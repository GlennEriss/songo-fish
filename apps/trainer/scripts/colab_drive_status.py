#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from colab_drive import detect_google_drive_root,ensure_layout,file_is_stable,sha256
def check(path,checksum):
 present=path.is_file();cp=checksum.is_file();valid=False
 if present and cp:
  try:valid=checksum.read_text().strip().split()[0]==sha256(path)
  except Exception:pass
 return present,path.stat().st_size if present else None,cp,valid
def main():
 p=argparse.ArgumentParser();p.add_argument("--experiment",required=True);p.add_argument("--drive-root");a=p.parse_args();root=detect_google_drive_root(a.drive_root);layout=ensure_layout(root);inp=layout["inputs"]/f"{a.experiment}_inputs.tar.gz";res=layout["exports"]/f"{a.experiment}_results.tar.gz";ip,isz,ic,iv=check(inp,Path(str(inp)+".sha256"));rp,rsz,rc,rv=check(res,Path(str(res)+".sha256"));local=Path("data/experiments")/("lot38_colab_compute" if a.experiment=="lot38" else a.experiment)
 out={"GOOGLE_DRIVE_DETECTED":"YES","DRIVE_ROOT":str(root),"INPUT_BUNDLE_PRESENT":ip,"INPUT_BUNDLE_SIZE":isz,"INPUT_CHECKSUM_VALID":iv,"EXPERIMENT_DIRECTORY_PRESENT":(layout["experiments"]/a.experiment).is_dir(),"RESULT_BUNDLE_PRESENT":rp,"RESULT_BUNDLE_STABLE":file_is_stable(res) if rp else False,"RESULT_CHECKSUM_PRESENT":rc,"RESULT_CHECKSUM_VALID":rv,"LOCAL_IMPORT_STATUS":"IMPORTED" if local.is_dir() and (local/"experiment_manifest.json").exists() else "NOT_IMPORTED","DRIVE_CLOUD_SYNC_STATUS":"UNKNOWN"};print(json.dumps(out,indent=2))
if __name__=="__main__":main()
