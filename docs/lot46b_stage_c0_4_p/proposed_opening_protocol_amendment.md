# LOT46B_STAGE_C0_OPENING_PROTOCOL_AMENDMENT_V1

**STATUS = PROPOSED — non approuvé.**

L'original C0 fige les tailles, budgets, appariements et statistiques, mais pas l'identité ni la construction des ouvertures. L'amendement propose un registre partagé de 400 états pour rendre les comparaisons cohérentes; ce choix est prospectif, pas historique.

Après approbation explicite des sources et SHA256, éliminer les états incomplets, terminaux, sans coup légal, sans split indépendant, les doublons physiques exacts et tout recouvrement exact avec training/validation. Ne pas canonicaliser les côtés. Classer les éligibles par `SHA256(seed || physical_state_fingerprint || source_record_id)`, retenir les 400 premiers et publier le registre canonique avec SHA256 avant tout jeu.

Le seed, les sources et leurs SHA256 restent à approuver. Aucun résultat de candidat, aucune Policy, Value ou MCTS ne participe à la sélection. Après gel, une donnée manquante provoque un arrêt; aucun remplacement silencieux.
