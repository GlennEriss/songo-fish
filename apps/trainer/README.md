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
