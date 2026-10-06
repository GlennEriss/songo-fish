"""Resolution explicite des chemins (section 12) : CLI > env > Colab > Drive desktop."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from colab_drive import DRIVE_PROJECT_NAME, detect_google_drive_root

from .config import EXPERIMENT_DIR_NAME

REPO_ROOT = Path(__file__).resolve().parents[4]
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")


@dataclass(frozen=True)
class ResolvedPaths:
    drive_root: Path | None
    drive_method: str
    project: Path | None
    lot41: Path
    lot42: Path
    lot43: Path
    output: Path
    exports: Path

    def as_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}


def resolve_drive_root(cli_value: str | None) -> tuple[Path | None, str]:
    """Retourne la racine Drive et la methode ; ``None`` si aucun Drive."""

    if cli_value:
        path = Path(cli_value).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"--drive-root does not exist: {path}")
        return path.resolve(), "CLI"
    env = os.environ.get("SONGO_DRIVE_ROOT")
    if env:
        path = Path(env).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"SONGO_DRIVE_ROOT does not exist: {path}")
        return path.resolve(), "ENV_SONGO_DRIVE_ROOT"
    if COLAB_DRIVE_ROOT.is_dir():
        return COLAB_DRIVE_ROOT, "COLAB_STANDARD"
    try:
        return detect_google_drive_root(None), "DRIVE_DESKTOP"
    except (FileNotFoundError, RuntimeError):
        return None, "NONE"


def resolve_paths(
    *,
    drive_root: str | None,
    output: Path | None,
    lot41: Path | None,
    lot42: Path | None,
    lot43: Path | None,
    exports: Path | None,
) -> ResolvedPaths:
    root, method = resolve_drive_root(drive_root)
    project = root / DRIVE_PROJECT_NAME if root is not None else None
    experiments = project / "experiments" if project is not None else REPO_ROOT / "data/experiments"

    def pick(explicit: Path | None, name: str) -> Path:
        return explicit.expanduser().resolve() if explicit is not None else experiments / name

    return ResolvedPaths(
        drive_root=root,
        drive_method=method,
        project=project,
        lot41=pick(lot41, "lot41_deep_mcts_convergence"),
        lot42=pick(lot42, "lot42_hard_position_65536"),
        lot43=pick(lot43, "lot43_multi_fidelity_teacher_protocol"),
        output=pick(output, EXPERIMENT_DIR_NAME),
        exports=exports.expanduser().resolve() if exports is not None else (project / "exports" if project is not None else REPO_ROOT / "data/exports"),
    )
