# trainer

Génération de dataset, entraînement du réseau, tournois. Consomme
`packages/songo_ai`.

**Runtime multi-provider** (septembre 2026) — `songo_ai.cloud`, voir
[`docs/trainer/provider-architecture.md`](../../docs/trainer/provider-architecture.md) :

- **`local` (défaut)** : tout tourne sur la machine — dev, tests,
  génération ≤ ~100k positions, **tout l'entraînement**, **tous les
  tournois**. `songo-cloud run {build,train,tournament} ...` (ou
  `python -m songo_ai.cloud ...`, ou `run_local.ps1` / `run_local.sh`).
- **`gcp`** : uniquement `run build`, pour une campagne d'annotation
  massive (1M+) ponctuelle après franchissement d'une porte de volume
  (§10.3). Provisionne une VM jetable qui exécute `build_100k_gcp.py` et
  s'auto-détruit.

Config : `songo.toml` (voir `songo.toml.example`) < env `SONGO_*`.
Historique : avant septembre 2026, génération sur GCP Compute Engine +
entraînement sur Colab/Drive ; les coûts GCP ont motivé le repli local.

**Reprise après arrêt brutal** : relancer un job avec les mêmes arguments
reprend où il s'était arrêté — `build` par le cache d'annotations (à la
position), `train` par `model_v<version>.resume.pt` (à l'époque),
`tournament` par un JSONL des parties jouées. Détails :
[`docs/trainer/provider-architecture.md`](../../docs/trainer/provider-architecture.md#reprise-après-un-arrêt-brutal).

## Pas à pas Windows — de zéro au modèle entraîné

PowerShell. Une fois le venv **activé** (étape 3), toutes les commandes
sont de simples `python ...` / `pip ...`. Si tu fermes le terminal,
réactive : `cd C:\dev\songo-fish ; .\.venv\Scripts\Activate.ps1`.

### 0. Sortir le projet de OneDrive

OneDrive casse le venv, corrompt le cache SQLite et essaie de synchroniser
des Go de dataset. Le projet doit être sur un chemin local simple.

```powershell
robocopy "$env:USERPROFILE\OneDrive\Documents\projets\songo-fish" "C:\dev\songo-fish" /E /MOVE
cd C:\dev\songo-fish
```

### 1. Code à jour

```powershell
git checkout dev
git pull origin dev
```

### 2. Créer l'environnement virtuel (Python 3.12+)

```powershell
py -3.12 --version
py -3.12 -m venv .venv
```

Si `py -3.12` échoue : installe Python depuis python.org (coche « Add
python.exe to PATH »), rouvre PowerShell, recommence. `py -3.13` marche
aussi.

### 3. Activer l'environnement virtuel

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
```

Le prompt doit maintenant commencer par `(.venv)`. `python` et `pip`
pointent désormais vers le venv.

### 4. pip patient (connexion instable) + torch CPU d'abord

```powershell
python -m pip config set global.timeout 120
python -m pip config set global.retries 10
python -m pip install --upgrade pip

python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

torch depuis l'index PyTorch = ~200 Mo (version CPU) au lieu de ~2,5 Go
(version CUDA, inutile ici). **Si le téléchargement coupe : relance la
même commande**, pip reprend où il en était.

### 5. Le reste des dépendances

```powershell
python -m pip install -e ".[dev,perf,train]"
```

torch est déjà satisfait, pip ne le re-télécharge pas. (`export`/onnx
volontairement omis — à ajouter plus tard pour le portage C#.)

> Si torch refuse de s'installer, tu peux quand même **générer le
> dataset** (torch ne sert qu'à l'entraînement) : `python -m pip install
> -e ".[perf]"` puis saute à l'étape 8.

### 6. Fichier de config

```powershell
@"
[runtime]
provider = "local"
num_workers = 6
device = "cpu"
"@ | Set-Content -Encoding ascii songo.toml
```

`num_workers = 6` = 6 positions annotées en parallèle (Phase 2). Baisser
si la RAM sature. `data_root` non renseigné ⇒ données dans `.\data\`.

### 7. Récupérer le champion actuel + vérifier

Copie **`data\checkpoints\`** depuis la machine précédente (`registry.json`
+ `model_v0.2.0.pt`, ~2 Mo) vers **`data\checkpoints\`** ici — nécessaire
pour le tournoi vs `champion` (étape 10).

```powershell
python -m pytest -q
python -m songo_ai.cloud config
```

Le tout premier `import` compile Numba (~20-30 s, une seule fois).

### 8. Pilote — mesure la vitesse réelle (~5-15 min)

```powershell
python -m songo_ai.cloud run build --positions 50 --seed 2026 --preset deep
```

La ligne `ETA … min` donne la durée du vrai run. Le preset `deep` utilise
déjà le professeur bidoua/territoire (rien à configurer, cf. plus bas).

### 9. Génération du dataset (long — laisser tourner)

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
python -m songo_ai.cloud run build --positions 100000 --seed 2026 --preset deep 2>&1 | Tee-Object -FilePath build_s2026_100k.log
```

**Si ça coupe : relance exactement la même commande.** Le cache
d'annotations (`data\datasets\dataset_s2026_100000\annotation_cache\`)
fait reprendre où c'était (log `reprise : X/100000 déjà en cache`).

### 10. Entraînement + tournois

```powershell
python -m songo_ai.cloud run train --version 0.3.0 --dataset datasets/dataset_s2026_100000 --epochs 150 --patience 15 2>&1 | Tee-Object -FilePath train_v0.3.0.log
python -m songo_ai.cloud run tournament --a 0.3.0 --b champion --games 200
python -m songo_ai.cloud run tournament --a 0.3.0 --b "minimax:4" --games 200
```

L'entraînement reprend aussi (`data\checkpoints\model_v0.3.0.resume.pt`).

### 11. Promotion — seulement si 0.3.0 gagne, après lecture des résultats

```powershell
python -c "from songo_ai.model import promote_version; promote_version('0.3.0')"
```

### Palier suivant

Après avoir vérifié que 0.3.0 ≥ champion, refaire 9→10 avec un volume
plus grand et un nouveau seed : `--positions 300000 --seed 2027`, puis
`--version 0.4.0 --dataset datasets/dataset_s2027_300000`.

## Pas à pas macOS / Linux — de zéro au modèle entraîné

Même provider (`local`), zsh/bash, **à lancer depuis la racine du dépôt**.

> **Si le dossier `.venv/` existe déjà** (machine qui a déjà servi) : saute
> l'étape 2 (et l'étape 4 si `python -m pytest -q` passe déjà). Fais les
> étapes 1, 3, 5, puis 6→10.

### 1. Code à jour

```bash
git checkout dev && git pull origin dev
```

### 2. Créer l'environnement virtuel (Python 3.12+)

Prend le premier interpréteur 3.12+ disponible (Homebrew fournit souvent
`python3.13`/`python3.14`, pas `python3.12`) :

```bash
python3.13 -m venv .venv || python3.12 -m venv .venv || python3.14 -m venv .venv || /opt/homebrew/bin/python3.13 -m venv .venv
```

Si aucun ne marche : `brew install python@3.13` (ou installeur python.org), rouvre le terminal, recommence.

### 3. Activer l'environnement virtuel

```bash
source .venv/bin/activate
python --version
```

Le prompt commence par `(.venv)` et `python --version` doit afficher
3.12+. `python` et `pip` pointent maintenant vers le venv.

> **Dans chaque nouveau terminal**, refais :
> ```bash
> cd /Users/glenneriss/Documents/projets/songo && source .venv/bin/activate
> ```

### 4. pip patient (connexion instable) + dépendances

```bash
python -m pip config set global.timeout 120
python -m pip config set global.retries 10
python -m pip install --upgrade pip
python -m pip install -e ".[dev,perf,train]"
```

Sur macOS, `torch` est déjà en version CPU/MPS (pas de CUDA à télécharger),
donc pas besoin d'index séparé. Si un téléchargement coupe, relance la
commande — pip reprend.

> Générer le dataset sans torch : `python -m pip install -e ".[perf]"`
> puis saute à l'étape 8 (torch ne sert qu'à l'entraînement).

### 5. Fichier de config

```bash
cat > songo.toml <<'EOF'
[runtime]
provider = "local"
num_workers = 6
device = "cpu"
EOF
```

`num_workers` = (cœurs physiques − 1). `data_root` non renseigné ⇒ données
dans `./data/`. `device = "cpu"` recommandé même avec un GPU Apple (MPS
n'accélère pas ce petit réseau).

### 6. Vérification

```bash
python -m pytest -q
python -m songo_ai.cloud config
```

### 7. Pilote — mesure la vitesse réelle

```bash
python -m songo_ai.cloud run build --positions 50 --seed 2026 --preset deep
```

### 8. Génération du dataset (long — `caffeinate` empêche la veille)

```bash
caffeinate -is python -m songo_ai.cloud run build \
  --positions 1000000 --seed 2026 --preset deep 2>&1 | tee build_s2026_100k.log
```

**Si ça coupe : relance exactement la même commande** — reprise via le
cache d'annotations (`data/datasets/dataset_s2026_100000/annotation_cache/`).

### 9. Entraînement + tournois

```bash
python -m songo_ai.cloud run train --version 0.3.0 \
  --dataset datasets/dataset_s2026_100000 --epochs 150 --patience 15 2>&1 | tee train_v0.3.0.log
python -m songo_ai.cloud run tournament --a 0.3.0 --b champion --games 200
python -m songo_ai.cloud run tournament --a 0.3.0 --b minimax:4 --games 200
```

### 10. Promotion — seulement si 0.3.0 gagne, après revue

```bash
python -c "from songo_ai.model import promote_version; promote_version('0.3.0')"
```

### Raccourci

`apps/trainer/scripts/run_local.sh run build --positions 100000 --seed 2026 --preset deep`
fait l'activation du venv + le lancement en une commande.

## Scripts

- `run_local.ps1` / `run_local.sh` — raccourci machine : active le venv et lance `python -m songo_ai.cloud` en provider `local`
- `bench_rules.py`, `bench_search.py`, `bench_fast_vs_reference.py` — benchmarks moteur (annexe C du plan)
- `build_100k_gcp.py` — raccourci `songo-cloud run build --preset deep` ; toujours appelé par `gcp_startup_script.sh` sur la VM (y écrit `/tmp/dataset_v002_<N>/`)
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
- `build_ordered_real_moves.py` — reconstruit, à partir des exports Firebase bruts (`data/real_matches/brutes/`, non versionnés, contiennent des données personnelles), le dataset **privacy-safe** de vrais coups humains prouvables (`data/real_matches/match_moves_v1.jsonl`, lui versionné) : ne garde que les transitions vérifiables par le moteur legacy (rejeu serveur exact, ou trajectoire courte à chemin unique), exclut bots/faux joueurs, n'écrit jamais d'identifiant/nom de joueur.
- `annotate_real_matches.py` — annote avec **notre** prof les positions de ces vraies parties humaines (`data/real_matches/match_moves_v1.jsonl`, 575 positions distinctes) pour les intégrer au dataset d'entraînement. On ne garde PAS le coup humain comme cible (pas une référence fiable) : seule la position est reprise, le prof produit best_action/action_values/WDL comme pour les autres datasets. Même sharding (`--shard-index`/`--num-shards`) et même `--cache-dir` frais obligatoire que les deux scripts ci-dessus.
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

## Entraîner (local ou Colab)

Local :

```bash
songo-cloud run train --version 0.3.0 --dataset datasets/dataset_v006 --epochs 150
songo-cloud run tournament --a 0.3.0 --b champion --games 200
# promotion explicite après revue :
python -c "from songo_ai.model import promote_version; promote_version('0.3.0')"
```

`colab_train.ipynb` : même chose sur Colab — monte Google Drive
(`SONGO_DATA_ROOT=<Drive>/SongoFish`), clone le dépôt, `LocalProvider`.
`File > Open notebook > GitHub` en pointant sur ce dépôt.

## Lancer une campagne d'annotation massive sur GCP

```bash
SONGO_PROVIDER=gcp songo-cloud run build --positions 1000000 --seed 789 --preset deep
```

`GcpProvider` empaquette le code, crée la VM (startup script,
`--max-run-duration`, auto-DELETE), et rend la commande de monitoring. La
VM exécute `build_100k_gcp.py` (= `LocalProvider` sur la VM) et pousse la
release vers GCS. Récupérer : `songo-cloud pull datasets/...`.

Équivalent manuel (inchangé) :

```bash
SONGO_GCS_BUCKET=gs://... apps/trainer/scripts/package_for_gcp.sh
gcloud compute instances create songo-dataset-XXX \
  --project songo-model-ai --zone us-central1-a \
  --machine-type c2d-highcpu-32 --image-family ubuntu-2404-lts-amd64 \
  --image-project ubuntu-os-cloud --boot-disk-size 50GB \
  --provisioning-model STANDARD --max-run-duration 43200s \
  --instance-termination-action DELETE --scopes cloud-platform \
  --metadata-from-file startup-script=apps/trainer/scripts/gcp_startup_script.sh \
  --metadata num-positions=100000
```
