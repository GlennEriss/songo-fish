#!/usr/bin/env python3
"""Modele pour tout futur entrainement : from scratch sur un dataset donne,
enregistrement automatique dans le registre de versions, puis tournoi
contre le champion actuel ET contre des agents de reference (section
10.2/10.3). Copier/adapter ce script (nouvelle version, nouveau dataset)
plutot que de reprendre train_and_eval_110k.py, qui reste l'archive de la
version 0.1.0.

La promotion en champion N'EST JAMAIS AUTOMATIQUE : ce script affiche les
resultats et laisse la decision (promote_version(...)) a un appel separe,
apres revue humaine (section 10.3 : "interdire toute promotion en presence
de coups illegaux ou de regression tactique")."""

from __future__ import annotations

import argparse
from pathlib import Path

from songo_ai.evaluation import play_match
from songo_ai.generation import make_shallow_search_agent, random_agent
from songo_ai.model import get_champion, load_model, make_network_agent, train_and_register

# opening_random_plies>0 : indispensable des que l'un des agents compares
# est deterministe, sinon "N parties" degenere en 2 parties repetees
# (cf. songo_ai.evaluation.tournament, avertissement en tete de module).
OPENING_RANDOM_PLIES = 6
TOURNAMENT_GAMES = 200


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help='ex: "0.2.0"')
    parser.add_argument("--dataset-dir", required=True, type=Path, help="dossier contenant train/val/test.jsonl + manifest.json")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    manifest = train_and_register(
        version=args.version,
        train_shard=args.dataset_dir / "train.jsonl",
        val_shard=args.dataset_dir / "val.jsonl",
        dataset_manifest_path=args.dataset_dir / "manifest.json",
        notes=args.notes,
    )
    print(f"=== Version {manifest.version} entrainee ===")
    print(f"epoques={manifest.metrics['epochs_run']} meilleure_epoque={manifest.metrics['best_epoch']}")
    print(f"val_loss={manifest.metrics['val_loss']:.4f} val_top1={manifest.metrics['val_policy_top1']:.3f}")

    checkpoint_path = Path("data/checkpoints") / f"model_v{args.version}.pt"
    model = load_model(checkpoint_path, dropout=manifest.architecture["dropout"])
    challenger_agent = make_network_agent(model, temperature=0.15)

    print()
    print("=== Tournoi vs agents de reference ===")
    for label, opponent in [
        ("aleatoire", random_agent),
        ("minimax profondeur 1", make_shallow_search_agent(max_depth=1, max_nodes=5_000, max_time_s=0.5)),
        ("minimax profondeur 4", make_shallow_search_agent(max_depth=4, max_nodes=50_000, max_time_s=1.0)),
    ]:
        result = play_match(
            challenger_agent, opponent, num_games=TOURNAMENT_GAMES, seed=hash(args.version) % 10_000,
            opening_random_plies=OPENING_RANDOM_PLIES,
        )
        print(
            f"vs {label}: score={result.score_a:.1f}/{result.games} ({result.win_rate_a*100:.1f}%, "
            f"IC95 [{result.ci_low*100:.1f}%, {result.ci_high*100:.1f}%], "
            f"{result.distinct_games} parties distinctes sur {result.games})"
        )

    champion = get_champion()
    if champion is not None and champion["version"] != args.version:
        print()
        print(f"=== Tournoi vs champion actuel (v{champion['version']}) ===")
        champion_model = load_model(Path(champion["checkpoint_path"]), dropout=champion["architecture"]["dropout"])
        champion_agent = make_network_agent(champion_model, temperature=0.15)
        result = play_match(
            challenger_agent, champion_agent, num_games=TOURNAMENT_GAMES, seed=hash(args.version) % 10_000,
            opening_random_plies=OPENING_RANDOM_PLIES,
        )
        print(
            f"challenger v{args.version} vs champion v{champion['version']}: "
            f"score={result.score_a:.1f}/{result.games} ({result.win_rate_a*100:.1f}%, "
            f"IC95 [{result.ci_low*100:.1f}%, {result.ci_high*100:.1f}%])"
        )
        print("Promotion : appeler explicitement promote_version(...) apres revue -- jamais automatique.")


if __name__ == "__main__":
    main()
