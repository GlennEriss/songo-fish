# Colab real execution guide

R1 autorise seulement le contrôle sûr :

```bash
python apps/trainer/scripts/run_lot46b_stage_c0_4_real.py \
  --registry-path docs/lot46b_stage_c0_4_p6/opening_registry_400.json \
  --checkpoint-root /content/songo-fish \
  --output-dir /content/lot46b-c04 \
  --drive-output /content/drive/MyDrive/songo-ai/experiments/lot46b_stage_c0_4_smoke \
  --preflight-only
```

Tests synthétiques :

```bash
python -m pytest -q packages/songo_ai/tests/test_stage_c0_4_r1.py
```

La commande R2 avec `--execute` n'est volontairement pas donnée comme commande exécutable pendant R1. Une autorisation de production liée au manifeste devra d'abord être installée et vérifiée.
