# Docs — trainer

Documentation spécifique au pipeline de génération/entraînement (au-delà du
plan directeur général, qui reste à la racine de `docs/`).

## Évaluation : magasins + territoire sûr (bidoua/Yinda)

Ajout de juillet 2026, suite à un retour de joueurs Songo pros : le
moteur jouait "glouton" (meilleure capture immédiate) au lieu de
construire une accumulation patiente dans une case ("grenier"/Yinda,
terminologie du livre *Le jeu de Songo*, S. Mbarga Owona, chapitre 4 —
voir `docs/livre songo Owona.pdf`) pour une capture future bien plus
importante — exactement la stratégie que les pros utilisent pour le
battre.

**Le fait exploitable** : une case ne perd jamais de graines sauf quand
son propriétaire la joue lui-même (un semis adverse qui y arrive ne fait
qu'ajouter une graine, jamais en retirer), et la capture ne s'applique
qu'aux cases à 2-4 graines au moment de l'arrivée. Donc une case à **5
graines ou plus est définitivement à l'abri** de la capture tant qu'elle
n'est pas jouée — c'est le début opérationnel d'un grenier, vérifiable
directement sur le plateau.

**Où c'est implémenté** :
- `packages/songo_ai/search/negamax.py` : `safe_territory()` (somme des
  graines des cases à 5+ d'un camp), `SAFE_ACCUMULATION_THRESHOLD = 5`,
  `SAFE_ACCUMULATION_WEIGHT = 3.0` (poids volontairement plus faible que
  le magasin, `10.0` — un grenier reste à jouer, pas encore un gain
  acquis). `default_evaluate()` (heuristique de secours, sans réseau)
  ajoute ce bonus à la différence de magasins pour toute position non
  terminale.
- `packages/songo_ai/hybrid/network_eval.py` : `make_network_evaluate()`
  ajoute le **même** bonus au score du réseau (`lean * EVAL_SCALE`). Le
  réseau champion actuel (v0.2.0) a été entraîné sur des labels générés
  **avant** ce correctif — sa tête WDL n'a pas encore appris cette notion
  elle-même, donc l'ajout explicite comble l'écart immédiatement, sans
  attendre une régénération de dataset + réentraînement. À retirer/réduire
  une fois qu'une future version, entraînée sur des labels
  post-correctif, démontre qu'elle l'a intériorisé (mesurable par
  tournoi).
- `packages/songo_ai/teachers/deep_teacher.py` (le professeur qui annote
  les datasets) : n'a rien à changer — il ne passe jamais `evaluate_fn` à
  `negamax_search`, donc il hérite automatiquement du bonus territoire
  dès qu'il atteint son horizon de recherche sans avoir trouvé la
  capture réelle. C'est là que le gain se propage vers les **futurs**
  datasets/ré-annotations, pas seulement vers la recherche brute. Gardé
  par un test de régression
  (`test_teacher_never_overrides_the_territory_aware_default_evaluate`).

**Vérifié en tournoi** (recherche pure, sans réseau, même profondeur,
`packages/songo_ai/evaluation/tournament.py`) : la version avec le bonus
territoire gagne 43 parties sur 60 contre la version sans (3 nulles),
IC95% [61,9%-83,5%] — effet réel, pas du bruit statistique sur un petit
échantillon.

**Portage C#** : voir `docs/integration_csharp_model_recherche.md` §3bis
pour la formule exacte à répliquer côté moteur de jeu.
