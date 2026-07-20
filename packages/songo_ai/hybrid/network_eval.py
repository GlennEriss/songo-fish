"""Evaluation de feuille + ordonnancement des coups bases sur le reseau
entraine (section 8.1), pour brancher dans songo_ai.search.negamax sans
dupliquer la recherche.

- Evaluation de feuille : tete WDL (win - loss) + bonus territoire sur
  (greniers/Yinda, cf. songo_ai.search.negamax.safe_territory), a la meme
  echelle que l'heuristique provisoire qu'elle remplace (default_evaluate
  = diff. magasins x10 + territoire*3 + mobilite -- cf. EVAL_SCALE). Le
  bonus territoire est ajoute explicitement ici (pas seulement laisse a la
  charge du reseau) car le champion actuel (v0.2.0) a ete entraine sur des
  labels generes AVANT le correctif bidoua/Yinda (juillet 2026, retour de
  joueurs pros : le moteur jouait "glouton", sans construire de grenier) --
  sa tete WDL n'a donc pas encore appris cette notion elle-meme. Ce bonus
  explicite comble l'ecart immediatement, sans attendre une regeneration
  de dataset + reentrainement. A retirer (ou reduire) une fois qu'une
  future version du reseau, entrainee sur des labels post-correctif,
  demontre qu'elle l'a interiorise (mesurable par tournoi, cf.
  songo_ai.evaluation.play_match).
  Les positions terminales restent evaluees par default_evaluate telles
  quelles (diff x1000, deja territoire-aware via final_score_with_territory) :
  leur magnitude doit toujours dominer l'evaluation du reseau, qui elle ne
  s'applique qu'aux feuilles non terminales.
- Ordonnancement : tete policy, pour explorer les coups les plus
  prometteurs en premier (meilleures coupures alpha-beta).

Les deux fonctions retrouvent le meme etat canonique que celui vu a
l'entrainement (songo_ai.dataset.schema.canonicalize_board) : le reseau n'a
jamais vu autre chose que "mon" plateau vu du joueur au trait.

Cout : mesure a profondeur 12 (voir hybrid/songofish.py), ~61% du temps
passait dans la reconstruction des features (legal_mask, canonicalisation)
via SongoLegacyGame -- moteur de reference en Python pur, pas concu pour
une boucle chaude. `game` ici est donc un `FastSongoGame` (Numba, meme
interface, meme resultat garanti par test differentiel section 4.3) des
que la recherche est en cours ; le type hint `SongoLegacyGame` documente
juste l'interface duck-typee requise (board/turn/legal_mask/zobrist_hash),
comme deja le cas pour EvaluateFn/PriorityFn dans search/negamax.py.

Cache : le passage avant PyTorch reste le second poste de cout (~75
microsecondes/appel). Un cache par hash Zobrist (les transpositions
ramenent les memes positions dans l'arbre) partage entre les deux tetes
fait qu'une position unique n'est jamais evaluee deux fois -- mais a
grande profondeur les transpositions sont rares (~1-2% de hits mesures),
l'essentiel du gain vient du cote FastSongoGame, pas du cache."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from songo_ai.dataset.schema import canonicalize_board
from songo_ai.model.features import observation_features
from songo_ai.model.network import SongoNet
from songo_ai.search.negamax import SAFE_ACCUMULATION_WEIGHT, EvaluateFn, PriorityFn, default_evaluate, safe_territory
from songo_ai.songo.rules import SongoLegacyGame, opponent, side_range

EVAL_SCALE = 300.0  # meme ordre de grandeur que default_evaluate (diff*10 + territoire*3, cf. _WDL_PLACEHOLDER_K)

# Purge simple du cache au-dela de cette taille (une partie entiere en
# consomme une fraction ; la purge complete evite une gestion LRU sans
# rapport avec le gain).
_CACHE_MAX_ENTRIES = 200_000


def _features_for(game: SongoLegacyGame) -> torch.Tensor:
    # tolist() : conversion numpy native (C), plus rapide qu'une
    # comprehension Python element par element -- utile ici car `game` est
    # generalement un FastSongoGame (board = numpy array).
    board = game.board.tolist() if hasattr(game.board, "tolist") else list(game.board)
    state = canonicalize_board(tuple(board), game.turn)
    legal_mask = game.legal_mask()
    return torch.tensor(observation_features(state, legal_mask), dtype=torch.float32).unsqueeze(0)


class NetworkCache:
    """Une passe avant par position unique (cle = hash Zobrist, qui couvre
    plateau + joueur au trait) : sert a la fois la policy (ordonnancement)
    et le lean WDL (evaluation de feuille)."""

    def __init__(self, model: SongoNet) -> None:
        model.eval()
        self._model = model
        self._entries: Dict[int, Tuple[List[float], float]] = {}

    def outputs(self, game: SongoLegacyGame) -> Tuple[List[float], float]:
        key = game.zobrist_hash()
        hit = self._entries.get(key)
        if hit is None:
            with torch.no_grad():
                policy_logits, wdl_logits, _ = self._model(_features_for(game))
            policy = torch.softmax(policy_logits, dim=-1).squeeze(0).tolist()
            wdl = torch.softmax(wdl_logits, dim=-1).squeeze(0)
            lean = (wdl[0] - wdl[2]).item()
            if len(self._entries) >= _CACHE_MAX_ENTRIES:
                self._entries.clear()
            hit = (policy, lean)
            self._entries[key] = hit
        return hit


def make_network_evaluate(model: SongoNet, cache: NetworkCache | None = None) -> EvaluateFn:
    net_cache = cache if cache is not None else NetworkCache(model)

    def evaluate(game: SongoLegacyGame, perspective: int) -> float:
        if game.finished:
            # Une position terminale doit toujours dominer l'evaluation du
            # reseau, quel que soit le signe -- on garde l'heuristique
            # existante pour ce cas, inchangee.
            return default_evaluate(game, perspective)

        # win/draw/loss, du point de vue du joueur au trait (canonicalize_board) ;
        # en negamax, perspective == game.turn a ce point (convention respectee
        # par negamax_search/iterative_deepening), donc "lean" est deja du point
        # de vue de `perspective`. Meme invariant reutilise ci-dessous pour le
        # bonus territoire : "mon" cote est celui de `perspective`.
        _, lean = net_cache.outputs(game)
        own_start, _ = side_range(perspective)
        opp_start, _ = side_range(opponent(perspective))
        territory_diff = safe_territory(game.board, own_start) - safe_territory(game.board, opp_start)
        return lean * EVAL_SCALE + territory_diff * SAFE_ACCUMULATION_WEIGHT

    return evaluate


def make_network_priority(model: SongoNet, cache: NetworkCache | None = None) -> PriorityFn:
    net_cache = cache if cache is not None else NetworkCache(model)

    def priority(game: SongoLegacyGame, legal_actions: List[int]) -> Dict[int, float]:
        policy, _ = net_cache.outputs(game)
        return {a: policy[a] for a in legal_actions}

    return priority
