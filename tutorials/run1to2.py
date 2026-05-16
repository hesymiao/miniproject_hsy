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
    adata1_path="/bigdat2/user/hesy/mini_project/datasets/after_process/multifeature_rep1.h5ad",
    adata2_path="/bigdat2/user/hesy/mini_project/datasets/after_process/multifeature_rep2.h5ad",
    device=DEVICE,
    epochs=500,
    num_layers=2,
    hidden_dim=256,
    latent_dim=128,
    spot_batch_size=1024,
    lr=1e-3,
    weight_decay=1e-5,
    seed=42,
    rand_keep_prob=0.8,#随机掩码20%
    comp_focus_min_prob=0.1,#归一化区间下界
    comp_focus_max_prob=0.5,#上界
    comp_focus_temperature=2.0,#sigmoid的时候的比例
    comp_focus_center=0.55,#sigmoid中心
    self_weight=0.5,#有private指导的恢复表达谱的权重
    stable_weight=1.0,#跨切片重建用的哪个，稳定的表达
    invariance_weight=0.4,#让掩码后的特征和不掩码的比较像，增强encoder提取稳定信息能力
    eval_spot_knn=8,#评估的时候的那个KNN，和spatialex用的一样
    val_spot_frac=0.0,#本来想设早停，发现没啥用
    early_stop_patience=0,
    private_dropout=0.5,
    output_dir=str(VERSION_DIR / "results" / "v40_1to2"),
    result_csv=str(VERSION_DIR / "1to2.csv"),
    use_spot_count_feature=True,
    use_resgcn=True,
    num_views=3,#切了几种
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
