# R2B Colab validation guide

R2B reste non exécutant.

1. Monter Drive et confirmer `/content/drive/MyDrive`.
2. Cloner `https://github.com/GlennEriss/songo-fish.git`, checkout du commit R2A publié et installer avec `pip install --no-deps -e`.
3. Préparer sous un même `--checkpoint-root` les chemins attendus par le CLI, sans modifier les fichiers sources.
4. Vérifier les hashes, le registre et l'autorisation avec le runner sans `--execute`, en fournissant `--authorization-file docs/lot46b_stage_c0_4_r2a/authorization_record.json`.
5. Vérifier CPU, CUDA éventuel, SQLite et lecture Drive.
6. Ne pas fournir `--execute` ni `--single-runner-confirmed` pendant R2B.

Le CLI réel et sa sortie `SAFE_PREFLIGHT_ONLY` constituent la commande de validation ; les chemins exacts doivent être résolus après montage effectif de Drive.
