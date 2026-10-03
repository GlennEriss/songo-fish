# Workflow Google Drive — Colab Songo

## Principe

Le dépôt Git reste la source de vérité du code. Google Drive contient les
entrées lourdes et les résultats persistants. Colab demeure un environnement
d'exécution éphémère et ne reçoit aucun serveur SSH, tunnel ou daemon distant.

La racine logique partagée est `songo-ai/` :

```text
songo-ai/
├── inputs/
├── experiments/
├── exports/
└── manifests/
```

Sur macOS, le bridge détecte Google Drive for desktop sous les emplacements
standards (`~/Library/CloudStorage/GoogleDrive*` et `/Volumes`). Dans Colab,
la même racine logique est `/content/drive/MyDrive/songo-ai/`.

## Préparer les entrées depuis le Mac

```bash
PYTHONPATH=packages:apps/trainer/scripts .venv/bin/python \
  apps/trainer/scripts/prepare_colab_inputs.py \
  --experiment lot38 \
  --sync-to-drive
```

La priorité de résolution du Drive est : `--drive-root`, puis
`SONGO_DRIVE_ROOT`, puis auto-détection. En présence de plusieurs Drives,
l'utilisateur doit choisir explicitement. La copie utilise un fichier
temporaire, un renommage atomique et une vérification taille/SHA256.

Le statut `DRIVE_LOCAL_COPY_VALID=YES` confirme seulement la copie locale dans
le filesystem Drive. La synchronisation réseau reste assurée par Google Drive
for desktop et son état est rapporté `UNKNOWN` s'il n'est pas vérifiable.

## Vérifier l'état

```bash
PYTHONPATH=packages:apps/trainer/scripts .venv/bin/python \
  apps/trainer/scripts/colab_drive_status.py --experiment lot38
```

## Exécuter dans Colab

Ouvrir `notebooks/songo_colab_compute_benchmark.ipynb`, lancer les cellules et
autoriser le montage Google Drive lorsque Google le demande. Le notebook lit
`inputs/lot38_inputs.tar.gz`, écrit les JSON sous
`experiments/lot38_colab_compute/`, puis crée :

```text
exports/lot38_results.tar.gz
exports/lot38_results.tar.gz.sha256
```

## Importer automatiquement sur le Mac

```bash
PYTHONPATH=packages:apps/trainer/scripts .venv/bin/python \
  apps/trainer/scripts/import_remote_experiment.py \
  --experiment lot38 \
  --from-drive \
  --wait-for-drive \
  --timeout 3600
```

Le script attend l'existence du bundle et de son checksum, vérifie que le
fichier est stable, valide SHA256, le manifest, le commit Git et les
fingerprints moteur/modèle, puis importe. Toute incompatibilité interrompt
explicitement l'opération. Aucun credential Google supplémentaire n'est
nécessaire : Drive for desktop et le montage Colab gèrent l'authentification.

## Intervention humaine restante

L'utilisateur doit seulement ouvrir/lancer le notebook Colab et autoriser le
montage Drive dans Colab lorsque Google le demande. Le reste du bridge local
est automatisable depuis le terminal lorsque Google Drive est monté.
