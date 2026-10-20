# Diagnostic du test distribué Lot45

Échec initial : le test attend les codes `[0, 0, 137]`, mais observe `[0, 0, 0]`. Deux réexécutions isolées passent respectivement en 14.05 s et 17.53 s.

`proc_worker` appelle correctement `os._exit(137)` dans le callback `on_claimed`. Cependant, les trois processus sont démarrés sans barrière; rien ne garantit que `w-crash` réserve un shard avant que `w-a` et `w-b` n’épuisent les douze shards. S’il ne réserve rien, son callback n’est jamais appelé et il termine normalement.

`REGRESSION_FAILURE_REPRODUCED = NO` lors du diagnostic borné. Classification : `FLAKY_TEST_PROCESS_SCHEDULING_RACE`. Cause la plus probable et directement supportée par le code : absence de synchronisation garantissant l’exercice du chemin de crash. Aucune régression de production n’est établie. Action : ticket de test séparé pour rendre le scénario déterministe; ne pas modifier la production, masquer ou désactiver le test dans P4.
