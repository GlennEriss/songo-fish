# Colab preexecution guide

Monter Drive, cloner le commit rapporté, installer le projet sans remplacer PyTorch CUDA, puis exécuter uniquement :

```bash
python apps/trainer/scripts/prepare_lot46b_stage_c0_4.py \
  --repo /content/songo-fish \
  --output /content/drive/MyDrive/songo-ai/experiments/lot46b_stage_c0_4_preexecution \
  --registry-path /content/songo-fish/docs/lot46b_stage_c0_4_p6/opening_registry_400.json \
  --unconstrained-checkpoint /content/drive/MyDrive/songo-ai/experiments/lot46b_b2/LOT46B_B2_STAGE_B_DEEP_UNCONSTRAINED_001/durable_checkpoints/checkpoint-00005000.pt \
  --device auto --dry-run --max-games 2
```

Le flag `--dry-run` est obligatoire. Ce programme n’importe aucun exécuteur de jeu et ne peut lancer ni partie ni MCTS. Vérifier les SHA publiés après copie Drive; ne jamais écraser P6.
