#!/usr/bin/env python3
"""Construit et synchronise les inputs d'une expérience Colab Songo."""
import argparse,json,tarfile,time
from pathlib import Path
from colab_drive import atomic_copy_verified,detect_google_drive_root,ensure_layout,sha256
from run_srn_colab_benchmark import git_commit,pool_fingerprints,engine_fingerprint
LOT38_FILES=(Path("data/experiments/lot35_generator_pool/g4_champion_identity.json"),Path("data/experiments/lot34r_g4_retry/checkpoints/pool/step-06000.pt"),Path("data/experiments/lot34r_g4_retry/checkpoints/pool/step-12000.pt"))
LOT39_POSITION_FILE=Path("data/colab_bridge/lot39_benchmark_positions.json")
LOT40_REFERENCE_FILES=(Path("data/colab_bridge/lot40_reference/lot39_scaling_256.json"),Path("data/colab_bridge/lot40_reference/lot39_decision.json"))
def main():
 p=argparse.ArgumentParser();p.add_argument("--experiment",required=True);p.add_argument("--sync-to-drive",action="store_true");p.add_argument("--drive-root");p.add_argument("--output-dir",type=Path,default=Path("data/colab_bridge"));a=p.parse_args()
 if a.experiment not in ("lot38","lot39","lot40"):raise ValueError("supported experiments: lot38, lot39, lot40")
 experiment_files=LOT38_FILES if a.experiment=="lot38" else LOT38_FILES+(LOT39_POSITION_FILE,) if a.experiment=="lot39" else LOT38_FILES+(LOT39_POSITION_FILE,)+LOT40_REFERENCE_FILES
 for f in experiment_files:
  if not f.is_file():raise FileNotFoundError(f)
 a.output_dir.mkdir(parents=True,exist_ok=True);bundle=a.output_dir/f"{a.experiment}_inputs.tar.gz";manifest_path=a.output_dir/f"{a.experiment}_inputs_manifest.json"
 manifest={"experiment_id":a.experiment,"git_commit":git_commit(),"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"model_fingerprints":pool_fingerprints(),"engine_fingerprint":engine_fingerprint(),"files":{str(f):sha256(f) for f in experiment_files}}
 manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
 with tarfile.open(bundle,"w:gz") as tf:
  for f in experiment_files:tf.add(f,arcname=str(f))
  tf.add(manifest_path,arcname=f"{a.experiment}_input_manifest.json")
 checksum=sha256(bundle);(bundle.with_suffix(bundle.suffix+".sha256")).write_text(f"{checksum}  {bundle.name}\n")
 result={"LOCAL_BUNDLE_CREATED":"YES","local_bundle":str(bundle),"sha256":checksum,"DRIVE_DETECTED":"NO","DRIVE_COPY_COMPLETED":"NO","DRIVE_COPY_CHECKSUM_VALID":"NO","DRIVE_CLOUD_SYNC_STATUS":"UNKNOWN"}
 if a.sync_to_drive:
  root=detect_google_drive_root(a.drive_root);layout=ensure_layout(root);copy=atomic_copy_verified(bundle,layout["inputs"]/bundle.name);atomic_copy_verified(manifest_path,layout["manifests"]/manifest_path.name);atomic_copy_verified(bundle.with_suffix(bundle.suffix+".sha256"),layout["inputs"]/(bundle.name+".sha256"));result.update({"DRIVE_DETECTED":"YES","drive_project_root":str(layout["project"]),"DRIVE_COPY_COMPLETED":"YES","DRIVE_COPY_CHECKSUM_VALID":"YES" if copy["valid"] else "NO","DRIVE_LOCAL_COPY_VALID":"YES" if copy["valid"] else "NO"})
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
