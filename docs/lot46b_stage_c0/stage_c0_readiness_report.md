# Lot46B Stage C0 — rapport de readiness

## Conclusion

La preregistration est prête, mais C1 n'est pas exécutable scientifiquement en l'état. `SongoRelationalNetwork` partage `node_encoder`, `global_encoder`, les blocs relationnels et `state_fusion` entre les deux sorties. Charger `policy_mlp` G5 et `value_mlp` G4R ne maintiendrait donc pas `V_G4R`. Aucun contournement approximatif n'est autorisé.

L'arène existante fournit agents indépendants, légalité moteur, résultats terminaux, côtés inversés, ouvertures appariées, seeds dérivées, bruit racine désactivé, température 0 et bootstrap par ouverture. Elle impose budgets/c_puct égaux dans une comparaison appariée. Il manque un runner durable C1/C2 avec identité des checkpoints, manifest complet, reprise/déduplication et orchestration asymétrique C2.

Le moteur demeure `SongoLegacyGame`, autorité unique pour 14 fosses, 70 graines, légalité, semis, captures et terminalité. Aucun candidat ne reçoit d'état privilégié. Minimax-Bidoua reste un adversaire externe figé (`70f3…334c`), jamais une source de labels.

## Alternatives à examiner ultérieurement

1. Comparer les modèles complets G4R/standard/unconstrained à budget égal, en déclarant explicitement que Policy et Value varient ensemble.
2. Concevoir dans un lot séparé une interface à encodeur/value véritablement figés, puis la revalider depuis zéro; aucune modification d'architecture n'est faite ici.

## Plan Colab futur

Restaurer B2/B3 dans des répertoires immuables, vérifier les deux SHA256 et chaque manifest, figer le commit, vérifier CUDA/GPU/versions, charger chaque checkpoint avec contrôle de formes et forward non-jeu. Estimer le temps uniquement depuis un benchmark autorisé. Écrire chaque résultat de façon atomique avec clé unique; au resume, rejeter changement de commit/config/checkpoint et ignorer uniquement les clés déjà validées. Exporter via staging, manifest de hashes, renommage atomique. L'entrée C1 doit exiger une autorisation explicite et refuser par défaut.

**CONTROLLED STOP : aucun match, entraînement, self-play, MCTS ou promotion.**
