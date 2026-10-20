# LOT46B_STAGE_C0_OPENING_PROTOCOL_AMENDMENT_V1

Statut : `PROPOSED`. Cet amendement prospectif ne réécrit pas la préinscription C0 historique, dont le registre d'ouvertures est absent. Il exige une approbation humaine explicite liée au SHA-256 des octets JSON de l'amendement.

La source proposée est `LOT45_D_REAL_HUMAN_SUBSET`. Le membre gzip gelé a le SHA-256 `49eb0f83265a763d06f7b45032d994227208f4149ad0c9fb67b6209fd8289667`; la source originale Git vérifiée a le SHA-256 `17d9ad19921cdbf1591eb54f6c40aa5b710376f6e903491dd39affc605a5ed18`. L'origine humaine externe demeure une limitation déclarée.

L'éligibilité exige un état complet, non terminal, avec coup légal, une empreinte physique unique, et l'absence des ensembles gelés d'entraînement et de validation. Sur 491 états uniques, 51 recouvrements d'entraînement sont exclus; les 994 états B3 sont raccordés à 100 % et leur intersection avec les 440 candidats restants est nulle.

Le registre proposé est partagé entre les six comparaisons C1. Après approbation seulement, `SHA256_RANK_V1` classe les candidats par SHA-256 de la concaténation exacte : seed entier non signé en décimal ASCII sans zéro initial, empreinte d'état en 64 caractères hexadécimaux minuscules ASCII, puis identifiant source exact en UTF-8. Tri croissant du digest; égalité départagée par empreinte d'état puis octets UTF-8 de l'identifiant. Les 400 premiers seraient sérialisés en JSON canonique (clés triées, séparateurs compacts, ASCII, NaN interdit), puis gelés par SHA-256. Aucun résultat de candidat n'entre dans ce calcul.

C1 reste fixé à six comparaisons, 400 clusters appariés et inversion des côtés, soit 800 parties par comparaison et 4800 au total. C2 conserve la marge de non-infériorité `-0.05`. Checkpoints, budgets MCTS, endpoints, tailles et règles statistiques sont inchangés.

Aucune sélection, aucun identifiant final, aucun registre et aucune partie ne sont produits par cet amendement. Toute modification des octets de l'amendement invalide son hash et exige une nouvelle approbation.
