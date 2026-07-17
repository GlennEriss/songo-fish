"""Evaluation de feuille + ordonnancement des coups bases sur le reseau
entraine (section 8.1), pour brancher dans songo_ai.search.negamax sans
dupliquer la recherche.

- Evaluation de feuille : tete WDL (win - loss), a la meme echelle que
  l'heuristique provisoire qu'elle remplace (songo_ai.search.negamax.
  default_evaluate = diff. magasins x10 + mobilite -- cf. EVAL_SCALE).
  Les positions terminales restent evaluees par default_evaluate telles
  quelles (diff x1000) : leur magnitude doit toujours dominer l'evaluation
  du reseau, qui elle ne s'applique qu'aux feuilles non terminales.
- Ordonnancement : tete policy, pour explorer les coups les plus
  prometteurs en premier (meilleures coupures alpha-beta).

Les deux fonctions retrouvent le meme etat canonique que celui vu a
l'entrainement (songo_ai.dataset.schema.canonicalize_board) : le reseau n'a
jamais vu autre chose que "mon" plateau vu du joueur au trait."""

from __future__ import annotations

from typing import Dict, List

import torch

from songo_ai.dataset.schema import canonicalize_board
from songo_ai.model.features import observation_features
from songo_ai.model.network import SongoNet
from songo_ai.search.negamax import EvaluateFn, PriorityFn, default_evaluate
from songo_ai.songo.rules import SongoLegacyGame

EVAL_SCALE = 300.0  # meme ordre de grandeur que default_evaluate (diff*10, cf. _WDL_PLACEHOLDER_K)


def _features_for(game: SongoLegacyGame) -> torch.Tensor:
    state = canonicalize_board(tuple(int(v) for v in game.board), game.turn)
    legal_mask = game.legal_mask()
    return torch.tensor(observation_features(state, legal_mask), dtype=torch.float32).unsqueeze(0)


def make_network_evaluate(model: SongoNet) -> EvaluateFn:
    model.eval()

    def evaluate(game: SongoLegacyGame, perspective: int) -> float:
        if game.finished:
            # Une position terminale doit toujours dominer l'evaluation du
            # reseau, quel que soit le signe -- on garde l'heuristique
            # existante pour ce cas, inchangee.
            return default_evaluate(game, perspective)

        with torch.no_grad():
            _, wdl_logits, _ = model(_features_for(game))
            probs = torch.softmax(wdl_logits, dim=-1).squeeze(0)

        # win/draw/loss, du point de vue du joueur au trait (canonicalize_board) ;
        # en negamax, perspective == game.turn a ce point (convention respectee
        # par negamax_search/iterative_deepening), donc "lean" est deja du point
        # de vue de `perspective`.
        lean = (probs[0] - probs[2]).item()
        return lean * EVAL_SCALE

    return evaluate


def make_network_priority(model: SongoNet) -> PriorityFn:
    model.eval()

    def priority(game: SongoLegacyGame, legal_actions: List[int]) -> Dict[int, float]:
        with torch.no_grad():
            policy_logits, _, _ = model(_features_for(game))
            probs = torch.softmax(policy_logits, dim=-1).squeeze(0)
        return {a: probs[a].item() for a in legal_actions}

    return priority
