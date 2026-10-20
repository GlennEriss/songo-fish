# Colab preexecution guide

Monter Drive, cloner `dev`, vérifier que `HEAD` descend de `433e1e462d430768fdd9592ee61a3e0b82111b3b`, puis installer le projet avec `pip install -e '.[train,dev]'` sans réinstaller PyTorch. Exécuter ensuite uniquement :

```bash
python apps/trainer/scripts/validate_lot46b_stage_c0_4_colab.py \
  --repo /content/songo-fish \
  --drive-root /content/drive/MyDrive/songo-ai \
  --device auto
```

Le validateur découvre le checkpoint exclusivement par son chemin historique et son SHA256, puis appelle lui-même l’entry point strict avec `--dry-run --max-games 2`. L’inférence synthétique est un chemin distinct. Chaque exécution publie dans un répertoire `attempt-<UTC>-<commit>` inédit et relit l’inventaire SHA256 depuis Drive. Aucun exécuteur de jeu ni MCTS n’est importé par le dry-run.
