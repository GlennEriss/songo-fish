# Production wiring

Le CLI construit un manifeste stable lié au baseline R1, valide registre, amendement, trois hashes de checkpoints et autorisation, sélectionne AUTO/CUDA/CPU, charge ensuite seulement les adaptateurs réels, puis délègue deux scheduled-game IDs à `production_control_flow`.

Chaque tentative est distincte, chaque résultat est transactionnel et chaque commit est suivi d'un snapshot restauré et vérifié. La reprise ne sélectionne que les scheduled games absents de `results`.
