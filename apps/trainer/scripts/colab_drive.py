"""Bridge filesystem générique entre le dépôt Songo et Google Drive Desktop."""
from __future__ import annotations
import hashlib, os, shutil, time
from dataclasses import dataclass
from pathlib import Path

DRIVE_PROJECT_NAME="songo-ai"

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()

def _valid_roots()->list[Path]:
 roots=[];cloud=Path.home()/"Library/CloudStorage"
 patterns=[(cloud,"GoogleDrive*"),(Path("/Volumes"),"GoogleDrive*")]
 for parent,pattern in patterns:
  if not parent.is_dir():continue
  for account in parent.glob(pattern):
   for name in ("My Drive","Mon Drive"):
    candidate=account/name
    if candidate.is_dir():roots.append(candidate.resolve())
   if account.is_dir() and account.parent==Path("/Volumes"):roots.append(account.resolve())
 return sorted(set(roots))

def detect_google_drive_root(explicit:str|Path|None=None)->Path:
 if explicit:
  path=Path(explicit).expanduser().resolve()
  if not path.is_dir():raise FileNotFoundError(f"explicit Google Drive root is invalid: {path}")
  return path
 env=os.environ.get("SONGO_DRIVE_ROOT")
 if env:return detect_google_drive_root(env)
 roots=_valid_roots()
 if not roots:raise FileNotFoundError("Google Drive for desktop root not detected")
 if len(roots)>1:raise RuntimeError("multiple Google Drive roots detected; use --drive-root or SONGO_DRIVE_ROOT")
 return roots[0]

def drive_project_root(root:Path)->Path:return root/DRIVE_PROJECT_NAME
def ensure_layout(root:Path)->dict[str,Path]:
 project=drive_project_root(root);result={name:project/name for name in ("inputs","experiments","exports","manifests")}
 for path in result.values():path.mkdir(parents=True,exist_ok=True)
 return {"project":project,**result}

def atomic_copy_verified(source:Path,destination:Path)->dict:
 destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+".part")
 shutil.copy2(source,temporary)
 if temporary.stat().st_size!=source.stat().st_size or sha256(temporary)!=sha256(source):
  temporary.unlink(missing_ok=True);raise RuntimeError("atomic Drive copy verification failed")
 os.replace(temporary,destination)
 return {"source_size":source.stat().st_size,"destination_size":destination.stat().st_size,"sha256":sha256(destination),"valid":sha256(destination)==sha256(source)}

def file_is_stable(path:Path,interval_s:float=2.0)->bool:
 if not path.is_file():return False
 first=(path.stat().st_size,path.stat().st_mtime_ns);time.sleep(interval_s)
 return path.is_file() and first==(path.stat().st_size,path.stat().st_mtime_ns)

def wait_for_files(bundle:Path,checksum:Path,timeout_s:float,poll_s:float=5.0)->None:
 deadline=time.monotonic()+timeout_s
 while time.monotonic()<deadline:
  if bundle.is_file() and checksum.is_file() and file_is_stable(bundle,min(2.0,poll_s)):
   expected=checksum.read_text().strip().split()[0]
   if expected==sha256(bundle):return
  time.sleep(poll_s)
 raise TimeoutError(f"Drive result not ready within {timeout_s}s")
