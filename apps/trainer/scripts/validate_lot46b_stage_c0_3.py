#!/usr/bin/env python3
"""Colab GPU validation only: archives, frozen checkpoints, synthetic SRN inference."""
from pathlib import Path
import argparse,json,subprocess,sys,time,torch
from songo_ai.evaluation.stage_c0_3 import verify_archive,validate_model_outputs,checkpoint_unchanged
from songo_ai.evaluation.arena_integration import cuda_readiness
from songo_ai.training.lot46a import ExperimentConfig,load_initial_model
from songo_ai.training.lot46b_b3 import _load_frozen
from songo_ai.model.srn_graph import SongoGraphBuilder
from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.songo.rules import PLAYER_ONE
from songo_ai.training.lot46 import sha256
def main():
 p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--drive-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 env=cuda_readiness(torch);assert env["status"]=="READY",env
 artifacts={"b2":verify_archive(a.drive_root/"exports/lot46b_b2_stage_b2_results.tar.gz","11e77ced813e8d71b2eb73dc2e8cb703dcc1b0dd5bba3b6c65d3c474879aea39"),"b3":verify_archive(a.drive_root/"exports/lot46b_b2_stage_b3_review.tar.gz","4c03c46c99bd5d1238dac15988bd0c02f577ca54d73a90a5239718c7d56c9809")}
 reports={};device=torch.device("cuda")
 for name,cfg_name in (("STANDARD","stage_b_deep_policy_standard.json"),("UNCONSTRAINED","stage_b_deep_unconstrained.json")):
  cfg=ExperimentConfig.load(a.repo/"configs/lot46b_b2"/cfg_name);model,path,manifest,payload,initial=_load_frozen(cfg,a.repo,device);before=sha256(path);model.eval();model.requires_grad_(False)
  graph=SongoGraphBuilder().build_batch([RawSongoState(tuple([5]*14+[0,0]),PLAYER_ONE)]).to(device)
  torch.manual_seed(123);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with torch.inference_mode():p1,v1=model(graph);p2,v2=model(graph)
  assert torch.equal(p1,p2) and torch.equal(v1,v2);contract=validate_model_outputs(p1,v1,1);checkpoint_unchanged(path,before)
  reports[name]={"checkpoint":str(path),"sha256":before,"step":manifest["step"],"contract":contract,"synthetic_inference_latency_s":time.perf_counter()-start,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"parameters_frozen":all(not x.requires_grad for x in model.parameters())}
 cfg=ExperimentConfig.load(a.repo/"configs/lot46b_b2/stage_b_deep_policy_standard.json");g4,audit=load_initial_model(cfg,a.repo,device);g4.eval();g4.requires_grad_(False)
 graph=SongoGraphBuilder().build_batch([RawSongoState(tuple([5]*14+[0,0]),PLAYER_ONE)]).to(device)
 with torch.inference_mode():gp,gv=g4(graph)
 reports["POOL_G4R"]={"policy_checkpoint":cfg.policy_checkpoint,"value_checkpoint":cfg.value_checkpoint,"contract":validate_model_outputs(gp,gv,1),"parameters_frozen":all(not x.requires_grad for x in g4.parameters()),"load_audit":audit}
 out={"environment":env,"artifacts":artifacts,"models":reports,"REAL_ARENA_VALIDATION":"NOT_RUN","MCTS_SEARCH":"NOT_RUN"}
 (a.output/"stage_c0_3_report.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));print("CONTROLLED STOP — synthetic inference only")
if __name__=="__main__":main()
