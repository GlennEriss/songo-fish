from pathlib import Path
import os,sys
import pytest
SCRIPTS=Path(__file__).resolve().parents[3]/"apps/trainer/scripts"
sys.path.insert(0,str(SCRIPTS))
from colab_drive import atomic_copy_verified,detect_google_drive_root,drive_project_root,ensure_layout,file_is_stable,sha256,wait_for_files
def test_explicit_drive_override(tmp_path):assert detect_google_drive_root(tmp_path)==tmp_path.resolve()
def test_environment_override(tmp_path,monkeypatch):monkeypatch.setenv("SONGO_DRIVE_ROOT",str(tmp_path));assert detect_google_drive_root()==tmp_path.resolve()
def test_missing_explicit_drive_fails(tmp_path):
 with pytest.raises(FileNotFoundError):detect_google_drive_root(tmp_path/"missing")
def test_project_layout_creation(tmp_path):
 layout=ensure_layout(tmp_path);assert layout["project"]==drive_project_root(tmp_path);assert all(layout[x].is_dir() for x in ("inputs","experiments","exports","manifests"))
def test_atomic_copy_and_checksum(tmp_path):
 src=tmp_path/"source";src.write_bytes(b"songo");dst=tmp_path/"drive"/"target";r=atomic_copy_verified(src,dst);assert r["valid"] and sha256(src)==sha256(dst);assert not dst.with_name(dst.name+".part").exists()
def test_stable_file_acceptance_and_unstable_rejection(tmp_path):
 p=tmp_path/"x";p.write_text("x");assert file_is_stable(p,.01)
 assert not file_is_stable(tmp_path/"absent",.01)
def test_wait_checksum_and_timeout(tmp_path):
 b=tmp_path/"lot_results.tar.gz";c=Path(str(b)+".sha256");b.write_bytes(b"ok");c.write_text(sha256(b)+"  lot_results.tar.gz\n");wait_for_files(b,c,.2,.01)
 c.write_text("bad\n")
 with pytest.raises(TimeoutError):wait_for_files(b,c,.05,.01)
def test_generic_naming_and_no_credentials():
 prepare=(SCRIPTS/"prepare_colab_inputs.py").read_text();status=(SCRIPTS/"colab_drive_status.py").read_text();imp=(SCRIPTS/"import_remote_experiment.py").read_text()
 assert 'f"{a.experiment}_inputs.tar.gz"' in prepare and 'f"{a.experiment}_results.tar.gz"' in status+imp
 joined=prepare+status+imp
 assert "oauth" not in joined.lower() and "google_token" not in joined.lower() and "credentials" not in joined.lower()
 assert ".backward(" not in joined and "torch.optim" not in joined
