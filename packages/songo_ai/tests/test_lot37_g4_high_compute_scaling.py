from pathlib import Path
S=(Path(__file__).resolve().parents[3]/"apps/trainer/scripts/run_srn_lot37.py").read_text()
def test_protocol():
 assert "BUDGETS=(512,1024,2048,4096)" in S and "GAMES=64" in S and "OPENINGS=32" in S
 assert '"baseline256_replayed":False' in S and '"BASELINE_SOURCE":"LOT36"' in S
def test_frozen_and_no_training():
 assert "exact_fingerprints" in S and "EXPECTED_MINIMAX" in S
 assert '"training_performed":False' in S and '"minimax_labels":False' in S and '"g5_training_performed":False' in S
def test_plateau_and_conditional_extension():
 assert "MATERIAL_GAIN=.03" in S and "plateau_pairs" in S
 assert '"rule":"S4096-S2048 >= .03"' in S and 'if not gate["authorized"]' in S
 assert '"no_mcts16384":True' in S
def test_paired_reproducible_schedule():
 assert '"paired":True' in S and '"side_swapped":True' in S
 assert '"shared_across_all_budgets":True' in S and "_pair" in S
