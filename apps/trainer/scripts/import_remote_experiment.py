#!/usr/bin/env python3
"""Importe un bundle expérimental distant après validation stricte."""
import argparse,json,tarfile,tempfile,time,shutil
from pathlib import Path
from run_srn_lot12 import sha256
from run_srn_colab_benchmark import engine_fingerprint,pool_fingerprints,git_commit
from colab_drive import detect_google_drive_root,ensure_layout,file_is_stable,wait_for_files
def main():
 p=argparse.ArgumentParser();p.add_argument("bundle",type=Path,nargs="?");p.add_argument("--experiment");p.add_argument("--from-drive",action="store_true");p.add_argument("--wait-for-drive",action="store_true");p.add_argument("--timeout",type=float,default=0);p.add_argument("--drive-root");p.add_argument("--destination",type=Path,default=Path("data/experiments"));p.add_argument("--allow-commit-mismatch",action="store_true");a=p.parse_args()
 if a.from_drive:
  if not a.experiment:raise ValueError("--experiment is required with --from-drive")
  root=detect_google_drive_root(a.drive_root);layout=ensure_layout(root);a.bundle=layout["exports"]/f"{a.experiment}_results.tar.gz";companion=Path(str(a.bundle)+".sha256")
  if a.wait_for_drive:
   if a.timeout<=0:raise ValueError("--wait-for-drive requires an explicit positive --timeout")
   wait_for_files(a.bundle,companion,a.timeout)
  else:
   if not a.bundle.is_file():raise FileNotFoundError(a.bundle)
   if not file_is_stable(a.bundle):raise RuntimeError("FILE_NOT_STABLE")
   if not companion.is_file():raise FileNotFoundError(companion)
   if companion.read_text().strip().split()[0]!=sha256(a.bundle):raise RuntimeError("Drive result checksum mismatch")
 if a.bundle is None:raise ValueError("provide a bundle or use --from-drive --experiment")
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
  local=pool_fingerprints()
  for k in ("policy_fingerprint","value_fingerprint","architecture_fingerprint"):
   if m["model_fingerprints"].get(k)!=local.get(k):raise RuntimeError(f"model fingerprint mismatch: {k}")
  for name,digest in m["artifact_checksums"].items():
   path=src/name
   if not path.exists() or sha256(path)!=digest:raise RuntimeError(f"artifact checksum mismatch: {name}")
  dest=a.destination/src.name
  if dest.exists():raise FileExistsError(f"destination already exists: {dest}; move or archive it before importing")
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(src,dest);print(dest)
if __name__=="__main__":main()
