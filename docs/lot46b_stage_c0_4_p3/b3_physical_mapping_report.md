# Récupération physique B3

Les 994/994 lignes B3 sont `EXACT_VERIFIED`; aucune n’est partielle, ambiguë, non résolue ou rejetée. Elles donnent 994 états physiques uniques, soit une couverture enregistrements et positions de 100 %.

Chaîne vérifiée : export B3 → `position_id` → empreinte de la ligne de validation Lot45 → état complet `{board, player_to_move}` → `RawSongoState` → empreinte physique canonique. Le code historique établit la première arête; l’unicité et la jointure exhaustive établissent la deuxième. Aucun logit Policy/Value n’est utilisé pour reconstruire un état.

L’empreinte historique Lot45 sérialise `board` et `player_to_move` sans le marqueur de schéma; l’empreinte de registre prospective ajoute le schéma et trie les clés. Elles ne sont pas comparées directement : la conversion passe toujours par l’état complet validé.
