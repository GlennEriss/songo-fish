"""Execution distribuee multi-Colab du Lot 45.

SHARD_COORDINATOR (``coordinator``) : baux transactionnels Firestore par shard,
jeton de fencing strictement croissant, verrous exclusifs de migration et de
finalisation.  WORKER_EXECUTOR (``worker``) : boucle reserver -> calculer ->
valider -> publier.  ARTIFACT_STORAGE (``storage``) : tentatives isolees
``attempts/{shard}/{lease}/`` sur Drive.  FINALIZER (``finalizer``) : assemble
uniquement les artefacts references par les enregistrements COMPLETED.

Drive n'est jamais utilise comme verrou : toute decision d'exclusivite passe
par une transaction Firestore.
"""
