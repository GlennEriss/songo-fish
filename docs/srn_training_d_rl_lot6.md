# Lot 6 — premier entrainement controle du SRN sur D_RL

Cette filiere est independante du MLP et du dataset teacher historiques. Elle
charge exclusivement le format versionne `D_RL`, conserve les etats physiques
et construit les features avec `SongoGraphBuilder` au moment du batching.

## Batch

Pour un batch de taille `B` :

- `node_features`: `[B,14,8]` ;
- `global_features`: `[B,5]` ;
- `legal_mask`: `[B,7]` ;
- `policy_target`: `[B,7]` ;
- `value_target`: `[B]` ;
- `value_mask`: `[B]`.

Une cible Value absente utilise un placeholder tensoriel nul, mais
`value_mask=False` garantit une contribution exactement nulle a la loss et au
gradient. Elle n'est jamais interpretee comme un nul Songo.

## Loss et metriques

```text
L_policy = -sum_a pi_target(a) log p_theta(a|S)
L_value  = moyenne (V_theta(S)-z)^2 sur value_mask=True seulement
L_total  = lambda_policy L_policy + lambda_value L_value
```

Les metriques Policy sont la cross-entropy distributionnelle, la KL et la
top-1 diagnostique. Les metriques Value sont MSE, MAE et exactitude du signe ;
une prediction est classee nulle lorsque sa valeur absolue est inferieure a
`value_sign_epsilon`.

## Split et reproductibilite

Le split est effectue par `metadata.game_id`. Une partie entiere appartient au
train ou a la validation, jamais aux deux. La seed controle l'initialisation,
le split et le shuffle. Le shuffle de chaque epoch utilise `seed + epoch`, ce
qui permet une reprise reproductible.

## Optimisation et checkpoints

La baseline utilise AdamW, sans penalite L2 manuelle, scheduler ou dropout. Le
clipping des gradients est configurable.

Chaque checkpoint contient les poids du modele, l'optimiseur, les deux
configurations, l'epoch, le pas global, la seed, le manifeste et hash du shard,
les game IDs du split, l'historique et la meilleure loss de validation. Deux
fichiers sont maintenus : `last_checkpoint.pt` et
`best_validation_checkpoint.pt`. Ce dernier terme ne designe jamais un
champion strategique.
