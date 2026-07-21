# trainer

Génération de dataset, entraînement du réseau, orchestration GCP + Google
Colab. Consomme `packages/songo_ai`.

**Deux providers de calcul, deux rôles distincts** (juillet 2026) :
- **GCP Compute Engine** : génération/annotation de dataset (le professeur
  tourne longtemps, CPU-bound, embarrassingly parallel — voir `build_100k_gcp.py`)
- **Google Colab** : entraînement du réseau (voir `colab_train.ipynb`).
  Datasets et checkpoints vivent sur **Google Drive** (pas sur Colab
  lui-même, éphémère), pas sur GCP — évite de garder une VM/un bucket GCP
  payant en continu pour un entraînement qui ne prend que quelques minutes.

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
- `build_curated_endgames.py` — annote avec un budget dédié (120s/position) des positions de fin de partie choisies à la main d'après `docs/III. Gestion des fins/` et `docs/II. Contre-Attaques 2/` (combinaisons jugées importantes par des joueurs expérimentés). Plutôt que d'extraire ces stratégies comme des règles codées en dur — vérifié coûteux et peu concluant même sur des positions à 9-10 graines (30-47 coups de profondeur, plusieurs millions de nœuds, toujours pas de résolution exacte) — le prof les annote avec sa propre recherche ; le réseau apprend une vérité ancrée dans la recherche, pas une heuristique humaine qu'on peine à vérifier soi-même
- `reannotate_external_dataset.py` — convertit et ré-annote avec **notre** prof de vraies parties reçues d'un pipeline externe (`songo-model-stockfish-for-google-collab`, format `.npz`, encodage de plateau différent — voir `songo_ai.dataset.external_import` pour le détail de la conversion validée). Plutôt que de garder leurs labels ("minimax insane" : profondeur fixe 22 mais seulement 1,2s/position, quasiment jamais atteinte en pratique), notre prof adaptatif produit une vérité tactique plus fiable.
- `reannotate_own_dataset.py` — même principe mais pour un dataset **déjà au format interne** (une release à nous, ex. `dataset_v003_110k`) : pas de conversion de plateau, juste une nouvelle passe d'annotation avec le professeur actuel. Sert notamment à propager un correctif du professeur (ex. bidoua/Yinda, juillet 2026) aux datasets déjà générés — sans ça, les labels déjà écrits restent gelés à l'ancienne version du professeur indéfiniment. **Attention cache** : le cache d'annotations est indexé par position + config du professeur, pas par version du code d'évaluation — toujours utiliser un `--cache-dir` neuf et clairement nommé (jamais réutiliser un cache d'une campagne antérieure à un correctif du professeur), sous peine de récupérer silencieusement d'anciennes annotations.
- Les deux scripts ci-dessus s'exécutent via `colab_reannotate_bidoua.ipynb` sur Colab (cache d'annotations sur Drive, résiste aux déconnexions) — **toujours piloter sur un sous-ensemble d'abord** : mesuré localement, ~15-16s/position pour le dataset externe (ancienne campagne) mais ~61s/position pour le 110k (juillet 2026, budget de 60s presque systématiquement épuisé) — deux chiffres non interchangeables, à revérifier sur l'environnement Colab réel avant de viser le volume complet (883k + 110k représente potentiellement plusieurs semaines de calcul selon le nombre de cœurs réellement disponibles)

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

## Entraîner sur Google Colab

`colab_train.ipynb` : monte Google Drive (dossier `SongoFish/` partagé au
préalable), clone le dépôt, entraîne, tourne le tournoi vs champion actuel
— tout sur Drive, rien sur GCP. Uploader ce notebook sur Colab (ou
`File > Open notebook > GitHub` en pointant sur ce dépôt) et l'exécuter
cellule par cellule.

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
