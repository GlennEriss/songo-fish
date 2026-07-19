#!/usr/bin/env python3
"""Exporte une version du registre vers ONNX (portable hors Python -- C#,
Unity/ML-Agents, etc. via Microsoft.ML.OnnxRuntime). Le reseau n'a aucune
operation exotique (Linear/ReLU uniquement) : l'export est direct, sans
adaptation particuliere.

Usage (necessite l'extra "export": .venv/bin/pip install -e ".[export]") :
    .venv/bin/python apps/trainer/scripts/export_onnx.py --version champion
    .venv/bin/python apps/trainer/scripts/export_onnx.py --version 0.2.0 --out data/checkpoints/model_v0.2.0.onnx

Voir docs/integration_csharp_model_recherche.md pour le format d'entree/
sortie et comment brancher le reseau exporte derriere une recherche
alpha-beta (C# ou tout autre langage hors Python).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from songo_ai.model import get_champion, load_model, load_registry
from songo_ai.model.features import FEATURE_SIZE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="champion", help='"champion" ou une version explicite, ex "0.2.0"')
    parser.add_argument("--out", type=Path, default=None, help="chemin de sortie .onnx (defaut : a cote du .pt)")
    args = parser.parse_args()

    entry = get_champion() if args.version == "champion" else load_registry()["versions"].get(args.version)
    if entry is None:
        raise ValueError(f"version inconnue: {args.version} (registre: data/checkpoints/registry.json)")

    model = load_model(
        Path(entry["checkpoint_path"]),
        dropout=entry["architecture"]["dropout"],
        width=entry["architecture"].get("width", 128),
        num_blocks=entry["architecture"].get("num_blocks", 3),
    )
    model.eval()

    out_path = args.out or Path(entry["checkpoint_path"]).with_suffix(".onnx")
    dummy_input = torch.zeros(1, FEATURE_SIZE, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["features"],
        output_names=["policy_logits", "wdl_logits", "q_values"],
        dynamic_axes={
            "features": {0: "batch"},
            "policy_logits": {0: "batch"},
            "wdl_logits": {0: "batch"},
            "q_values": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Version {entry['version']} exportee -> {out_path}")
    print(f"Entree attendue : tenseur float32 [batch, {FEATURE_SIZE}] (cf. docs/integration_csharp_model_recherche.md)")


if __name__ == "__main__":
    main()
