# Device fallback policy

`AUTO` préfère CUDA et retombe sur CPU avant une partie si CUDA est indisponible ou échoue à l'initialisation. `CPU` ne sonde pas CUDA. `CUDA` échoue explicitement si indisponible.

Une erreur CUDA pendant une partie interrompt la tentative. Une nouvelle tentative, avec un nouvel ID mais le même scheduled-game ID et la même ouverture, doit repartir sur CPU. Aucun résultat déjà commité n'est rejoué. Le budget reste 512.
