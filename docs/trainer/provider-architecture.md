# Runtime multi-provider : `local` (defaut) et `gcp`

Décision de conception, septembre 2026. Contexte : les coûts GCP étaient
devenus disproportionnés (entraînement **et** génération de dataset
tournaient sur GCP). Objectif : tout ramener sur une machine Windows
(18 Go RAM, CPU ~2,30 GHz ~4 cœurs, SSD 500 Go, un GPU), **sans supprimer**
la possibilité de repasser sur GCP pour une campagne massive ponctuelle.

C'est exactement la trajectoire prévue par le plan directeur §11.3
(« Local d'abord ») et la porte d'étape (« Locale d'abord ; GCP en option
après benchmark »). Cette migration ne fait que rendre `local` **effectif
par défaut** au lieu de rester théorique.

## Ce qui était couplé à GCP — et ce qui ne l'était pas

Le cœur `songo_ai` était **déjà** provider-agnostic : `build_dataset()`,
`train_and_register()`, `AnnotationCache`, `play_match()` ne prennent que
des chemins et un nombre de workers / un device. Le couplage vivait
uniquement dans la couche orchestration : chemins `gs://` en dur, cycle de
vie VM, boucle de synchronisation cache → GCS, notebooks Colab avec des
chemins `/content/drive/...`.

**Donc on n'abstrait pas le calcul. On abstrait deux choses :**

| Préoccupation | Abstraction | Fichier |
|---|---|---|
| Où vivent les artefacts (datasets, caches, checkpoints, registre) | `ArtifactStore` (Strategy) | `songo_ai/cloud/storage.py` |
| Où/comment un job s'exécute | `ComputeProvider` (Strategy + Template Method) | `songo_ai/cloud/providers.py` |
| Quel provider, quel dossier, combien de workers, quel device | `RuntimeConfig` | `songo_ai/cloud/config.py` |
| Quoi calculer (indépendant du provider) | `BuildSpec` / `TrainSpec` / `MatchSpec` | `songo_ai/cloud/jobs.py` |

## Résolution de la config

`defaults` < `songo.toml` (racine du dépôt, non versionné — voir
`songo.toml.example`) < variables d'env `SONGO_*`.

```
SONGO_PROVIDER      local | gcp           (défaut local)
SONGO_DATA_ROOT     dossier des artefacts (défaut <repo>/data)
SONGO_NUM_WORKERS   workers d'annotation  (défaut : moitié des cœurs logiques)
SONGO_DEVICE        cpu | cuda | auto     (défaut cpu)
SONGO_GCS_BUCKET / SONGO_GCS_PREFIX       (provider gcp)
```

`num_workers` par défaut = **moitié des cœurs**, pas `os.cpu_count()` :
chaque worker instancie son propre professeur avec une table de
transposition de plusieurs Mo. Sur la machine cible, viser 3–6. Le startup
script GCP, lui, réexporte `SONGO_NUM_WORKERS=$(nproc)` (une
`c2d-highcpu-32` a de la marge).

## Les deux providers

### `LocalProvider` — le quotidien

Exécute le job dans le process courant. Couvre : dev, tests, génération
≤ ~100k positions, **tout l'entraînement**, **tous les tournois**. Tout
vit sur le SSD (`data_root`) — **plus besoin de Google Drive**, qui ne
servait qu'à donner un stockage persistant à Colab (VM éphémère).

## Reprise après un arrêt brutal

Chaque job long reprend là où il s'est arrêté quand on le **relance avec
les mêmes arguments** (mêmes `--positions`/`--seed`, même `--version`,
mêmes agents). Rien de spécial à taper.

| Job | État de reprise | Granularité |
|---|---|---|
| `build` | `datasets/<nom>/annotation_cache/cache.db` (SQLite, commit après **chaque** position) | la position |
| `train` | `checkpoints/model_v<version>.resume.pt` (poids + optimiseur + historique, écrit après **chaque** époque, atomiquement) | l'époque |
| `tournament` | `tournaments/<a>_vs_<b>_...jsonl` (une ligne par partie jouée) | la partie |

- `build` : l'échantillonnage est déterministe (même seed → mêmes
  positions **et** mêmes `trajectory_id`), donc au redémarrage on
  recalcule le hash de chaque position et on ne redonne au pool que
  celles absentes du cache. Une reprise à 95 % ne recalcule que les 5 %
  restants. Le fichier de reprise n'est jamais supprimé (le cache est
  aussi un accélérateur pour les campagnes suivantes au même preset).
- `train` / `tournament` : le fichier de reprise est **supprimé à la fin
  d'un run complet**. S'il est encore là, c'est qu'un run a été
  interrompu.
- Sécurité : `train` ignore un `.resume.pt` dont l'architecture ne
  correspond pas ; `tournament` rejoue les parties d'un fichier tronqué
  et reprend proprement (chaque partie a son propre RNG `(seed, index)`,
  donc identique qu'il y ait eu reprise ou non).

Le `GcpProvider` garde en plus sa synchronisation continue du cache vers
GCS (`gcp_startup_script.sh`) : une VM tuée par `--max-run-duration` est
rattrapée par la VM suivante qui réhydrate le cache depuis le bucket.

### `GcpProvider` — la campagne massive ponctuelle

Ne gère **que** `run_build`. Provisionne une VM Compute Engine jetable
(`--max-run-duration`, `--instance-termination-action DELETE`), qui
exécute `build_100k_gcp.py` — c'est-à-dire le `LocalProvider` **sur la
VM** — puis pousse la release vers GCS et s'auto-détruit. Réutilise tels
quels `package_for_gcp.sh` et `gcp_startup_script.sh`, seul le bucket
devient paramétrable.

`run_train` / `run_tournament` lèvent `NotImplementedError` : le réseau
est minuscule (quelques minutes de CPU), payer du cloud pour ça n'a aucun
sens.

## Répartition recommandée

| Charge | Provider | Pourquoi |
|---|---|---|
| Dev, tests, benchmarks | local | — |
| Génération dataset ≤ ~100k | local | ~un week-end sur la machine cible |
| Tout l'entraînement | local | quelques minutes CPU |
| Tous les tournois | local | pur CPU |
| Annotation massive 1M+, après franchissement d'une porte de volume (§10.3) | gcp | 8–15× plus rapide, mais payant — donc seulement quand un gain est déjà prouvé au palier inférieur |

### Ordre de grandeur

Preset `deep` ≈ 6,5 s/position en moyenne.

- `c2d-highcpu-32` (~32 workers) : 100k ≈ **6 h**
- Machine cible (~6 workers, cœur ~1,5× plus lent) : 100k ≈ **~40–45 h**
- 10k local : une soirée. 1M local : impraticable → ligne GCP.

## Utilisation

```bash
# via l'entrypoint installé (project.scripts)
songo-cloud config
songo-cloud run build      --positions 10000 --seed 456 --preset deep
songo-cloud run train      --version 0.3.0 --dataset datasets/dataset_v006
songo-cloud run tournament --a champion --b minimax:8

# équivalent
python -m songo_ai.cloud run build --positions 10000 --seed 456

# raccourcis machine
apps/trainer/scripts/run_local.ps1 run build --positions 10000   # Windows
apps/trainer/scripts/run_local.sh  run build --positions 10000   # macOS/Linux
```

Presets professeur (`songo_ai/cloud/jobs.py`, source unique) :
`default` (5 s/pos, palier 10k) · `deep` (palier 100k) · `reannotate`
(premium, propagation d'un correctif).

Agents de tournoi : `vX.Y.Z` · `champion` · `random` · `minimax:<depth>`.

## Points d'attention Windows

1. **`ProcessPoolExecutor` = spawn** (pas fork). Tout entrypoint doit
   avoir `if __name__ == "__main__"`. Vérifié par
   `test_multiprocessing_entrypoints_are_guarded`.
2. **`build_dataset` pré-crée `cache.db`** dans le process principal avant
   de lancer le pool (sinon « database is locked » quand plusieurs workers
   initialisent WAL en même temps sur un cache neuf).
3. **numba (`perf`) est de fait obligatoire** : `fast_rules.py` l'importe
   sans garde. Wheels dispo pour CPython 3.12 Windows.
4. **`data_root` sur le SSD interne** — pas OneDrive / lecteur réseau
   (SQLite WAL ne les supporte pas).
5. **GPU** : `device="cpu"` reste le défaut recommandé. Le GPU ne
   redeviendra pertinent qu'après un changement d'architecture MAJOR.

## Limites connues

- Le préfixe GCS `pilots/` est encore codé en dur dans
  `apps/trainer/scripts/*.sh` (le changer aux deux endroits si besoin).
- `RuntimeConfig` suppose une install éditable (`pip install -e`) pour
  localiser la racine du dépôt.
- `GcpProvider.run_build` ne fait pas de polling (les jobs durent ~40 h) —
  il rend la commande de monitoring et le chemin de récupération.
