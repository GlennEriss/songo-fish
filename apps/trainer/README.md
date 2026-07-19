# trainer

Génération de dataset, entraînement du réseau, orchestration GCP. Consomme
`packages/songo_ai`.

## Scripts

- `bench_rules.py`, `bench_search.py`, `bench_fast_vs_reference.py` — benchmarks moteur (annexe C du plan)
- `build_100k_gcp.py` — génération/annotation à grande échelle (config profonde), destiné à tourner sur GCP
- `sync_cache.py` — snapshot à chaud du cache SQLite (utilisé par `gcp_startup_script.sh`)
- `train_overfit_10k.py` — surapprentissage volontaire (validation pipeline, étape 6)
- `train_and_eval_110k.py` — entraînement réel + tournoi contre agents de référence (étape 7)
- `package_for_gcp.sh` — empaquette le code et le pousse vers Cloud Storage
- `gcp_startup_script.sh` — script de démarrage de la VM (rehydratation cache, build, upload, arrêt auto)
- `register_existing_models.py` — enregistrement rétroactif ponctuel des deux premières versions (ne pas relancer)
- `train_new_version.py` — **modèle pour tout futur entraînement** : from scratch, enregistrement automatique, tournoi vs référence + champion actuel (`--version 0.2.0 --dataset-dir data/dataset_vXXX`)
- `export_onnx.py` — exporte une version du registre vers ONNX (portage hors Python, ex. C#/Unity via `Microsoft.ML.OnnxRuntime`) : `--version champion`. Voir `docs/integration_csharp_model_recherche.md` pour le format d'entrée/sortie du réseau et comment le brancher derrière une recherche alpha-beta

## Versioning des modèles (`packages/songo_ai/model/registry.py`)

**Chaque version = un entraînement complet from scratch sur un dataset donné, jamais un fine-tuning d'une version précédente.**

Pourquoi : le plan directeur organise déjà l'amélioration du modèle autour d'un
cycle champion/challenger avec porte de promotion statistique (section 10.2/10.3)
— un nouveau candidat est entraîné et comparé à l'ancien champion, pas fusionné
avec lui. Et le réseau est assez petit (quelques centaines de milliers de
paramètres, quelques minutes d'entraînement même sur 100k+ positions) pour que
"économiser du calcul en repartant de l'existant" ne soit jamais un argument
valable ici — ça compte pour de gros modèles, pas pour celui-ci.

Convention de version (semver, indépendante de la version du package `songo-ai`
dans `pyproject.toml`) :

- **MAJOR** : changement d'architecture (largeur, nombre de blocs, features d'entrée) — versions non comparables directement
- **MINOR** : nouvel entraînement sur un dataset plus grand/différent, même architecture — le cas normal à chaque palier de volume
- **PATCH** : même dataset, même architecture, autre run (seed, hyperparamètres) — itération fine

Versions actuelles :

| Version | Dataset | Statut | Notes |
|---|---|---|---|
| 0.0.1 | 10k (standard) | archivé | surapprentissage volontaire, validation pipeline (étape 6), pas un joueur |
| 0.1.0 | 110k (10k standard + 100k profond) | archivé | premier entraînement réel, généralisation validée, bat aléatoire/minimax profondeur 1 |
| 0.2.0 | 392k dédupliquées (110k + 290k, seed 789) | **champion** | val_top1=54,7% (vs 48,4% pour 0.1.0) ; bat le champion précédent v0.1.0 à 81,5% (200 parties) ; réseau seul (sans recherche) reste faible face à minimax profondeur 4 (7,8%), mais une fois branché dans SongoFish (réseau+recherche), bat un minimax profondeur 14/5s de réflexion |

Utilisation pour un nouvel entraînement :

```python
from songo_ai.model import train_and_register

manifest = train_and_register(
    version="0.2.0",
    train_shard=Path("data/dataset_vXXX/train.jsonl"),
    val_shard=Path("data/dataset_vXXX/val.jsonl"),
    dataset_manifest_path=Path("data/dataset_vXXX/manifest.json"),
    notes="...",
)
# promotion en champion seulement apres revue des metriques / tournoi vs le
# champion actuel (jamais automatique) :
from songo_ai.model import promote_version
promote_version("0.2.0")
```

## Lancer un job GCP

```bash
apps/trainer/scripts/package_for_gcp.sh
gcloud compute instances create songo-dataset-XXX \
  --project songo-model-ai --zone us-central1-a \
  --machine-type c2d-highcpu-32 --image-family ubuntu-2404-lts-amd64 \
  --image-project ubuntu-os-cloud --boot-disk-size 50GB \
  --provisioning-model STANDARD --max-run-duration 43200s \
  --instance-termination-action DELETE --scopes cloud-platform \
  --metadata-from-file startup-script=apps/trainer/scripts/gcp_startup_script.sh \
  --metadata num-positions=100000
```
