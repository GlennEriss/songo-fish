"""Runtime multi-provider (section 11 du plan : "Architecture GCP et
maitrise du budget" -- ici generalisee : `local` par defaut, `gcp` en
option).

- `config`    : `RuntimeConfig` (defaults < songo.toml < env `SONGO_*`)
- `storage`   : `ArtifactStore` -- `LocalStore` / `GcsStore`
- `jobs`      : `BuildSpec` / `TrainSpec` / `MatchSpec` + presets professeur
- `providers` : `LocalProvider` / `GcpProvider`

CLI : `python -m songo_ai.cloud run {build,merge,train,tournament} ...`
"""

from .config import GcpConfig, RuntimeConfig, load_config
from .jobs import TEACHER_PRESETS, BuildSpec, MatchSpec, MergeSpec, TrainSpec
from .providers import ComputeProvider, GcpProvider, LocalProvider, make_provider
from .storage import ArtifactStore, GcsStore, LocalStore, make_store

__all__ = [
    "GcpConfig",
    "RuntimeConfig",
    "load_config",
    "TEACHER_PRESETS",
    "BuildSpec",
    "MatchSpec",
    "MergeSpec",
    "TrainSpec",
    "ComputeProvider",
    "GcpProvider",
    "LocalProvider",
    "make_provider",
    "ArtifactStore",
    "GcsStore",
    "LocalStore",
    "make_store",
]
