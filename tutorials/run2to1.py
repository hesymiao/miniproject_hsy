import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.train_eval import TrainConfig, run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
VERSION_DIR = Path(__file__).resolve().parents[1]
DEVICE = 'cuda:0'


cfg = TrainConfig(
    adata1_path="/bigdat2/user/hesy/mini_project/datasets/after_process/multifeature_rep2.h5ad",
    adata2_path="/bigdat2/user/hesy/mini_project/datasets/after_process/multifeature_rep1.h5ad",
    device=DEVICE,
    epochs=600,
    num_layers=2,
    hidden_dim=256,
    latent_dim=128,
    spot_batch_size=1024,
    lr=1e-3,
    weight_decay=1e-5,
    seed=42,
    rand_keep_prob=0.8,
    comp_focus_min_prob=0.1,
    comp_focus_max_prob=0.5,
    comp_focus_temperature=2.0,
    comp_focus_center=0.55,
    self_weight=0.5,
    stable_weight=1.0,
    invariance_weight=0.4,
    eval_spot_knn=8,
    val_spot_frac=0.0,
    early_stop_patience=0,
    private_dropout=0.5,
    output_dir=str(VERSION_DIR / "results" / "v40_2to1"),
    result_csv=str(VERSION_DIR / "2to1.csv"),
    use_spot_count_feature=True,
    use_resgcn=True,
    num_views=3,
    view_mixer_layers=2,
    view_mixer_dropout=0.1,
    gcn_layers=2,
    gcn_knn=12,
    gcn_dropout=0.05,
    gcn_residual_scale=0.5,
    eval_cell_knn=7,
    eval_cell_ssim_sigma=0.01,
)

summary = run_experiment(cfg)
print(summary.tail(1).to_string(index=False))
