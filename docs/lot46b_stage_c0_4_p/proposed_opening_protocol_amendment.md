# LOT46B_STAGE_C0_OPENING_PROTOCOL_AMENDMENT_V1

**STATUS = PROPOSED — non approuvé.**

L'original C0 fige les tailles, budgets, appariements et statistiques, mais pas l'identité ni la construction des ouvertures. L'amendement propose un registre partagé de 400 états pour rendre les comparaisons cohérentes; ce choix est prospectif, pas historique.

Après approbation explicite des sources et SHA256, éliminer les états incomplets, terminaux, sans coup légal, sans split indépendant, les doublons physiques exacts et tout recouvrement exact avec training/validation. Ne pas canonicaliser les côtés. Classer les éligibles par `SHA256(seed || physical_state_fingerprint || source_record_id)`, retenir les 400 premiers et publier le registre canonique avec SHA256 avant tout jeu.

La qualification P2 recommande provisoirement le sous-ensemble `D_REAL_HUMAN` du conteneur Lot45 (SHA256 `49eb0f83265a763d06f7b45032d994227208f4149ad0c9fb67b6209fd8289667`) et propose le seed public `3538971373371044805`. Ils restent non approuvés tant que la fuite B3, indéterminable sans états physiques, n'est pas résolue. Aucun résultat de candidat, aucune Policy, Value ou MCTS ne participe à la sélection. Après gel, une donnée manquante provoque un arrêt; aucun remplacement silencieux.
