import io,tarfile
import pytest,torch
from songo_ai.evaluation.stage_c0_3 import *
from songo_ai.evaluation.arena_runner import *

def test_archive_hash_and_safe_members(tmp_path):
 p=tmp_path/"a.tar.gz"
 with tarfile.open(p,"w:gz") as t:
  x=tarfile.TarInfo("report.json");b=b"{}";x.size=len(b);t.addfile(x,io.BytesIO(b))
 assert verify_archive(p,file_sha256(p))["status"]=="PASS"
 with pytest.raises(ValueError):verify_archive(p,"0"*64)
def test_output_and_mask_contract():
 p=torch.zeros(2,7);v=torch.tensor([-.5,.5]);assert validate_model_outputs(p,v,2)["finite"]
 validate_legal_mask(p,torch.ones(2,7,dtype=torch.bool))
 with pytest.raises(ValueError):validate_model_outputs(p,torch.tensor([float("nan"),0]),2)
def test_checkpoint_immutability(tmp_path):
 p=tmp_path/"x";p.write_bytes(b"x");h=file_sha256(p);checkpoint_unchanged(p,h);p.write_bytes(b"y")
 with pytest.raises(ValueError):checkpoint_unchanged(p,h)
def test_frozen_c1_schedule():
 c=lambda n:Candidate(n,"x","a"*64,"c","srn","m","t",1)
 names=("G4","STD","UNC");pairs=((0,1),(0,2),(1,2));comps=[]
 for budget in (512,4096):
  for a,b in pairs:comps.append({"comparison_id":f"{names[a]}-{names[b]}-{budget}","candidate_a":names[a],"candidate_b":names[b],"budget_a":budget,"budget_b":budget})
 m=ExperimentManifest("C1","v1","c","e"*64,tuple(c(n) for n in names),tuple(comps),tuple(f"o{i:03}" for i in range(400)),7,{"margin":-0.05},"cuda","fp32",{})
 assert validate_c1_schedule(m)["games"]==4800
