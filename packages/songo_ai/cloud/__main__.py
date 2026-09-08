"""Point d'entree unique du runtime.

    python -m songo_ai.cloud run build      --positions 10000 --seed 456 --preset deep
    python -m songo_ai.cloud run train      --version 0.2.1 --dataset datasets/dataset_v005_400k
    python -m songo_ai.cloud run tournament --a v0.2.1 --b champion
    python -m songo_ai.cloud config          # affiche la config effective
    python -m songo_ai.cloud pull  datasets/dataset_v005_400k

Le provider (`local` / `gcp`) vient de `songo.toml` ou de `SONGO_PROVIDER`,
surchargeable par `--provider`.

Le bloc `if __name__ == "__main__"` est indispensable : `build_dataset`
lance un `ProcessPoolExecutor` et Windows demarre ses workers par *spawn*
(re-import du module principal), pas par *fork*.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .jobs import TEACHER_PRESETS, BuildSpec, MatchSpec, TrainSpec
from .providers import make_provider


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="songo_ai.cloud")
    parser.add_argument("--provider", choices=["local", "gcp"], help="surcharge songo.toml / SONGO_PROVIDER")
    parser.add_argument("--data-root", help="surcharge le dossier des artefacts")
    parser.add_argument("--workers", type=int, help="surcharge num_workers")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="affiche la config effective")

    p_pull = sub.add_parser("pull", help="rapatrie un artefact logique depuis le stockage de verite")
    p_pull.add_argument("logical")

    run = sub.add_parser("run", help="execute un job")
    jobs = run.add_subparsers(dest="job", required=True)

    b = jobs.add_parser("build", help="generer + annoter un dataset (les matchs)")
    b.add_argument("--positions", type=int, required=True)
    b.add_argument("--seed", type=int, default=456)
    b.add_argument("--preset", choices=sorted(TEACHER_PRESETS), default="deep")
    b.add_argument("--name", help="nom logique du dataset (defaut derive du volume + seed)")

    t = jobs.add_parser("train", help="entrainer une version du reseau from scratch")
    t.add_argument("--version", required=True)
    t.add_argument("--dataset", required=True, help="chemin logique, ex datasets/dataset_v005_400k")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--promote", action="store_true", help="deconseille : promouvoir apres revue via promote_version()")
    t.add_argument("--notes", default="")

    m = jobs.add_parser("tournament", help="tournoi entre deux agents")
    m.add_argument("--a", required=True, help='vX.Y.Z | champion | random | minimax:<depth>')
    m.add_argument("--b", required=True)
    m.add_argument("--games", type=int, default=200)
    m.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    config = load_config().with_overrides(
        provider=args.provider, data_root=args.data_root, num_workers=args.workers
    )

    if args.command == "config":
        _print({
            "provider": config.provider,
            "data_root": str(config.data_root),
            "num_workers": config.num_workers,
            "device": f"{config.device} -> {config.resolved_device()}",
            "gcp": vars(config.gcp),
        })
        return 0

    provider = make_provider(config)

    if args.command == "pull":
        _print({"pulled": str(provider.store.pull(args.logical))})
        return 0

    if args.job == "build":
        spec = BuildSpec(
            num_positions=args.positions, seed=args.seed,
            teacher_preset=args.preset, dataset_name=args.name,
        )
        _print(provider.run_build(spec))
    elif args.job == "train":
        spec = TrainSpec(
            version=args.version, dataset_name=args.dataset, epochs=args.epochs,
            early_stopping_patience=args.patience, promote=args.promote, notes=args.notes,
        )
        _print(provider.run_train(spec))
    elif args.job == "tournament":
        spec = MatchSpec(agent_a=args.a, agent_b=args.b, num_games=args.games, seed=args.seed)
        _print(provider.run_tournament(spec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
