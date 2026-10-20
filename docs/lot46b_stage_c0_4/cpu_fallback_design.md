# CPU fallback design

Le fallback porte sur les réseaux complets. Aucun échange isolé de tête Policy ou Value n’est permis (`POLICY_VALUE_COMPATIBILITY=BLOCKED_SHARED_ENCODER`). Les poids, l’architecture, les paramètres MCTS, la seed et l’ouverture ne changent pas.

La table SQLite `attempts` conserve device et erreur par tentative. La table `results`, indexée par l’identité scientifique de partie, conserve l’exactly-once. Ainsi, plusieurs tentatives techniques ne créent jamais plusieurs résultats.
