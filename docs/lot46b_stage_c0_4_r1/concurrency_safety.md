# Concurrency safety

Google Drive monté ne fournit pas de primitive atomique de coordination distribuée démontrée. R1 ne revendique donc aucun distributed lock.

La politique est fail-closed : une future R2 doit fournir une identité opérateur non vide et confirmer explicitement qu'aucune autre session n'est active. La reprise concurrente automatique est interdite. Le risque résiduel dépend du respect de cette discipline opératoire.
