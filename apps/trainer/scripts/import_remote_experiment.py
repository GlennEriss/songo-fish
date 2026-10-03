#!/usr/bin/env python3
"""Importe un bundle expérimental distant après validation stricte."""
import argparse,json,tarfile,tempfile
from pathlib import Path
from run_srn_lot12 import sha256
from run_srn_colab_benchmark import engine_fingerprint,exact_fingerprints,git_commit
def main():
 p=argparse.ArgumentParser();p.add_argument("bundle",type=Path);p.add_argument("--destination",type=Path,default=Path("data/experiments"));p.add_argument("--allow-commit-mismatch",action="store_true");a=p.parse_args()
 with tempfile.TemporaryDirectory() as td:
  root=Path(td)
  with tarfile.open(a.bundle,"r:gz") as tf:
   for m in tf.getmembers():
    if m.name.startswith("/") or ".." in Path(m.name).parts:raise RuntimeError("unsafe bundle path")
   tf.extractall(root)
  dirs=[x for x in root.iterdir() if x.is_dir()]
  if len(dirs)!=1:raise RuntimeError("bundle must contain exactly one experiment directory")
  src=dirs[0];mp=src/"experiment_manifest.json"
  if not mp.exists():raise RuntimeError("missing experiment_manifest.json")
  m=json.load(mp.open())
  if m.get("lot")!=38:raise RuntimeError("wrong lot")
  if m["git_commit"]!=git_commit() and not a.allow_commit_mismatch:raise RuntimeError("git commit mismatch")
  if m["engine_fingerprint"]!=engine_fingerprint():raise RuntimeError("engine fingerprint mismatch")
  local=exact_fingerprints()["POOL"]
  for k in ("policy_fingerprint","value_fingerprint","architecture_fingerprint"):
   if m["model_fingerprints"].get(k)!=local.get(k):raise RuntimeError(f"model fingerprint mismatch: {k}")
  for name,digest in m["artifact_checksums"].items():
   path=src/name
   if not path.exists() or sha256(path)!=digest:raise RuntimeError(f"artifact checksum mismatch: {name}")
  dest=a.destination/src.name
  if dest.exists():raise FileExistsError(f"destination already exists: {dest}")
  dest.parent.mkdir(parents=True,exist_ok=True);src.rename(dest);print(dest)
if __name__=="__main__":main()
