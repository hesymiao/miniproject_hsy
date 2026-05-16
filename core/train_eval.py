from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .data import SliceData, build_device_cache, build_spot_batch_from_cache, read_slice
from .model import InterventionStableGenerator


@dataclass
class TrainConfig:
    adata1_path: str
    adata2_path: str
    device: str = "cuda:3"
    epochs: int = 500
    num_views: int = 3
    num_layers: int = 3
    hidden_dim: int = 256
    latent_dim: int = 128
    spot_batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    rand_keep_prob: float = 0.7
    comp_focus_min_prob: float = 0.05
    comp_focus_max_prob: float = 0.35
    comp_focus_temperature: float = 8.0
    comp_focus_center: float = 0.6
    self_weight: float = 1.0
    stable_weight: float = 0.5
    invariance_weight: float = 0.1
    eval_spot_knn: int = 8
    val_spot_frac: float = 0.0
    early_stop_patience: int = 0
    private_dropout: float = 0.3
    output_dir: str = ""
    result_csv: str = ""
    use_spot_count_feature: bool = False
    use_resgcn: bool = False
    gcn_layers: int = 2
    gcn_knn: int = 8
    gcn_dropout: float = 0.1
    gcn_residual_scale: float = 0.2
    view_mixer_layers: int = 2
    view_mixer_dropout: float = 0.1
    eval_cell_knn: int = 7
    eval_cell_ssim_sigma: float = 0.01


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _domain_tensor(n: int, value: int, device: str) -> torch.Tensor:
    return torch.full((n,), value, dtype=torch.long, device=device) #长度为 n、每个元素都等于 value


def _mean_by_spot(cell_pred: torch.Tensor, local_spot_ids: torch.Tensor, num_spots: int) -> torch.Tensor:
    out = torch.zeros((num_spots, cell_pred.shape[1]), dtype=cell_pred.dtype, device=cell_pred.device)
    counts = torch.zeros(num_spots, dtype=cell_pred.dtype, device=cell_pred.device)
    out.index_add_(0, local_spot_ids, cell_pred)
    counts.index_add_(0, local_spot_ids, torch.ones_like(local_spot_ids, dtype=cell_pred.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(1)


def _append_spot_count(he: torch.Tensor, spot_count: torch.Tensor | None, cfg: TrainConfig) -> torch.Tensor:
    if cfg.use_spot_count_feature:
        if spot_count is None:
            raise ValueError("spot_count feature is enabled but missing from batch.")
        return torch.cat([he, spot_count], dim=1)
    return he


def _make_interventions(
    he: torch.Tensor,
    rand_keep_prob: float,
    comp_focus_min_prob: float,
    comp_focus_max_prob: float,
    comp_focus_temperature: float,
    comp_focus_center: float,
) -> tuple[torch.Tensor, ...]:
    rand_mask = (torch.rand_like(he) < rand_keep_prob).to(he.dtype)#随机掩码
    saliency = torch.sigmoid(comp_focus_temperature * (he - comp_focus_center))#根据 he 的数值大小手工算一个显著性分数
    #某维he明显大于comp_focus_center，saliency 接近 1，comp_focus_temperature 决定过渡有多陡
    focus_prob = comp_focus_min_prob + (comp_focus_max_prob - comp_focus_min_prob) * saliency #从 [0,1] 映射到一个概率区间,低值维度也不是绝对不进，只是概率更低
    focus_mask = (torch.rand_like(he) < focus_prob).to(he.dtype)
    empty_focus = focus_mask.sum(dim=1, keepdim=True) == 0
    if empty_focus.any():
        top_idx = he.argmax(dim=1, keepdim=True)
        focus_mask = focus_mask.scatter(1, top_idx, torch.ones_like(top_idx, dtype=he.dtype))#保底取一维一下
    context_mask = 1.0 - focus_mask
    return he, he * rand_mask, he * focus_mask, he * context_mask


def _slice_losses(outputs: dict[str, torch.Tensor], spot_target: torch.Tensor, local_spot_ids: torch.Tensor, num_spots: int, cfg: TrainConfig) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    spot_self = _mean_by_spot(outputs["pred_self"], local_spot_ids, num_spots)
    spot_stable = _mean_by_spot(outputs["pred_stable"], local_spot_ids, num_spots)

    self_loss = F.mse_loss(spot_self, spot_target)
    stable_loss = F.mse_loss(spot_stable, spot_target)
    inv_loss = (
        F.mse_loss(outputs["shared_orig"], outputs["shared_rand"])
        + F.mse_loss(outputs["shared_orig"], 0.5 * (outputs["shared_comp1"] + outputs["shared_comp2"]))
    )#鲁棒性约束，防止共享编码器过度依赖某些局部维度
    loss = (
        cfg.self_weight * self_loss
        + cfg.stable_weight * stable_loss
        + cfg.invariance_weight * inv_loss
    )
    stats = {
        "self": float(self_loss.detach().cpu()),
        "stable": float(stable_loss.detach().cpu()),
        "inv": float(inv_loss.detach().cpu()),
    }
    return loss, stats, {
        "spot_self": spot_self,
        "spot_stable": spot_stable,
    }


def _iter_spot_batches(num_spots: int, batch_size: int, device: str) -> list[torch.Tensor]:
    order = torch.randperm(num_spots, device=device, dtype=torch.long)
    return list(order.split(batch_size))


def _iter_spot_batches_from_ids(spot_ids: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    if spot_ids.numel() == 0:
        return []
    order = spot_ids[torch.randperm(spot_ids.shape[0], device=spot_ids.device, dtype=torch.long)]
    return list(order.split(batch_size))


def _evaluate_spot_loss(
    model: InterventionStableGenerator,
    cache: dict[str, torch.Tensor],
    spot_ids: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[float, float]:
    if spot_ids.numel() == 0:
        return float("nan"), float("nan")
    model.eval()
    losses = []
    self_losses = []
    with torch.no_grad():
        for batch_spot_ids in spot_ids.split(cfg.spot_batch_size):
            batch = build_spot_batch_from_cache(cache, batch_spot_ids)
            x1_orig, x1_rand, x1_comp1, x1_comp2 = _make_interventions(
                batch["he"],
                cfg.rand_keep_prob,
                cfg.comp_focus_min_prob,
                cfg.comp_focus_max_prob,
                cfg.comp_focus_temperature,
                cfg.comp_focus_center,
            )
            out = model(
                x_orig=x1_orig,
                x_rand=x1_rand,
                x_comp1=x1_comp1,
                x_comp2=x1_comp2,
                local_spot_ids=batch["local_spot_ids"],
                spot_adj=_build_spot_adj(batch["spot_spatial"], cfg.gcn_knn) if cfg.use_resgcn else None,
            )
            loss, stats, _ = _slice_losses(out, batch["spot_expr"], batch["local_spot_ids"], batch["num_spots"], cfg)
            losses.append(float(loss.detach().cpu()))
            self_losses.append(stats["self"])
    model.train()
    return float(np.mean(losses)), float(np.mean(self_losses))


def _build_spot_knn(coords: torch.Tensor, k: int) -> torch.Tensor:
    if coords.shape[0] <= 1:
        return torch.zeros((coords.shape[0], 1), dtype=torch.long, device=coords.device)
    k_eff = min(k + 1, coords.shape[0])
    dist = torch.cdist(coords, coords)
    idx = torch.topk(dist, k=k_eff, largest=False).indices
    return idx[:, 1:]


def _build_spot_adj(coords: torch.Tensor, k: int) -> torch.Tensor:
    n = coords.shape[0]
    if n <= 1:
        return torch.eye(n, dtype=coords.dtype, device=coords.device)
    knn_idx = _build_spot_knn(coords, k)
    adj = torch.zeros((n, n), dtype=coords.dtype, device=coords.device)
    src = torch.arange(n, device=coords.device, dtype=torch.long).unsqueeze(1).expand_as(knn_idx)
    adj[src.reshape(-1), knn_idx.reshape(-1)] = 1.0
    adj = torch.maximum(adj, adj.T)
    adj.fill_diagonal_(1.0)
    deg = adj.sum(dim=1)
    inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)


def _gpu_pcc(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = y_true - y_true.mean(dim=0, keepdim=True)
    y = y_pred - y_pred.mean(dim=0, keepdim=True)
    denom = torch.sqrt((x.pow(2).sum(dim=0) * y.pow(2).sum(dim=0)).clamp_min(1e-8))
    pcc = (x * y).sum(dim=0) / denom
    return pcc, pcc.mean()


def _gpu_cmd(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = y_true - y_true.mean(dim=0, keepdim=True)
    y = y_pred - y_pred.mean(dim=0, keepdim=True)
    x = x / x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    y = y / y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    corr_true = (x.T @ x) / max(x.shape[0], 1)
    corr_pred = (y.T @ y) / max(y.shape[0], 1)
    numerator = torch.trace(corr_pred @ corr_true)
    denominator = torch.linalg.norm(corr_pred) * torch.linalg.norm(corr_true)
    cmd = 1.0 - numerator / denominator.clamp_min(1e-8)
    return cmd.unsqueeze(0), cmd


def _gpu_graph_ssim(y_true: torch.Tensor, y_pred: torch.Tensor, knn_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nbr_true = y_true[knn_idx]
    nbr_pred = y_pred[knn_idx]
    ux = nbr_true.mean(dim=1)
    uy = nbr_pred.mean(dim=1)
    vx = nbr_true.var(dim=1, unbiased=False)
    vy = nbr_pred.var(dim=1, unbiased=False)
    vxy = ((nbr_true - ux.unsqueeze(1)) * (nbr_pred - uy.unsqueeze(1))).mean(dim=1)
    data_range = (y_true.max(dim=0).values - y_true.min(dim=0).values).clamp_min(1e-4)
    c1 = (0.01 * data_range).pow(2)
    c2 = (0.03 * data_range).pow(2)
    ssim = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux.pow(2) + uy.pow(2) + c1) * (vx + vy + c2))
    per_gene = ssim.mean(dim=0)
    return per_gene, per_gene.mean()


def _build_spatialex_style_knn_graph_gpu(
    coords: torch.Tensor,
    num_neighbors: int,
    sigma: float,
    query_chunk: int = 1024,
    ref_chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = coords.shape[0]
    k_eff = min(num_neighbors, n)
    if n == 0:
        raise ValueError("Cannot build KNN graph for an empty coordinate set.")
    if n == 1:
        idx = torch.zeros((1, 1), dtype=torch.long, device=coords.device)
        weight = torch.ones((1, 1), dtype=coords.dtype, device=coords.device)
        return idx, weight

    ref_norm = (coords * coords).sum(dim=1)
    all_idx = []
    all_w = []
    denom = max(2.0 * np.pi * sigma * sigma, 1e-12)
    query_chunk = max(1, min(query_chunk, n))
    ref_chunk = max(k_eff, min(ref_chunk, n))

    for start in range(0, n, query_chunk):
        end = min(start + query_chunk, n)
        q = coords[start:end]
        q_norm = (q * q).sum(dim=1, keepdim=True)
        best_d2 = None
        best_idx = None

        for ref_start in range(0, n, ref_chunk):
            ref_end = min(ref_start + ref_chunk, n)
            ref = coords[ref_start:ref_end]
            ref_norm_chunk = ref_norm[ref_start:ref_end]
            d2 = (q_norm + ref_norm_chunk.unsqueeze(0) - 2.0 * (q @ ref.T)).clamp_min_(0.0)
            vals, idx = torch.topk(
                d2,
                k=min(k_eff, ref.shape[0]),
                dim=1,
                largest=False,
                sorted=False,
            )
            idx = idx + ref_start
            if best_d2 is None:
                best_d2 = vals
                best_idx = idx
            else:
                merged_d2 = torch.cat([best_d2, vals], dim=1)
                merged_idx = torch.cat([best_idx, idx], dim=1)
                best_d2, top_pos = torch.topk(
                    merged_d2,
                    k=min(k_eff, merged_d2.shape[1]),
                    dim=1,
                    largest=False,
                    sorted=False,
                )
                best_idx = merged_idx.gather(1, top_pos)

        weights = torch.exp(-(best_d2 / 2.0) * (sigma * sigma)) / denom
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min_(1e-12)
        all_idx.append(best_idx)
        all_w.append(weights.to(dtype=coords.dtype))

    return torch.cat(all_idx, dim=0), torch.cat(all_w, dim=0)


def _spatialex_style_ssim_gpu(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_w: torch.Tensor,
    k1: float = 0.01,
    k2: float = 0.03,
    query_chunk: int = 512,
    gene_chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_range = (y_true.max() - y_true.min()).clamp_min(1e-12)
    c1 = (k1 * data_range).square()
    c2 = (k2 * data_range).square()
    c3 = ((k2 / np.sqrt(2.0)) * data_range).square()
    num_neighbor = nbr_idx.shape[1]
    cov_norm = float(num_neighbor) / float(max(num_neighbor - 1, 1) + 1e-6)
    gene_sum = torch.zeros(y_true.shape[1], dtype=torch.float32, device=y_true.device)
    query_chunk = max(1, min(query_chunk, y_true.shape[0]))
    gene_chunk = max(1, min(gene_chunk, y_true.shape[1]))

    for start in range(0, y_true.shape[0], query_chunk):
        end = min(start + query_chunk, y_true.shape[0])
        idx = nbr_idx[start:end]
        w = nbr_w[start:end].unsqueeze(-1)
        for g_start in range(0, y_true.shape[1], gene_chunk):
            g_end = min(g_start + gene_chunk, y_true.shape[1])
            x_n = y_true[idx, g_start:g_end]
            y_n = y_pred[idx, g_start:g_end]
            ux = (w * x_n).sum(dim=1)
            uy = (w * y_n).sum(dim=1)
            uxx = (w * (x_n * x_n)).sum(dim=1)
            uyy = (w * (y_n * y_n)).sum(dim=1)
            uxy = (w * (x_n * y_n)).sum(dim=1)
            vx = cov_norm * (uxx - ux * ux)
            vy = cov_norm * (uyy - uy * uy)
            vxy = cov_norm * (uxy - ux * uy)
            a1 = 2.0 * ux * uy + c1
            a2 = 2.0 * torch.sqrt((vx * vy).clamp_min_(0.0)) + c2
            a3 = vxy + c3
            b1 = ux * ux + uy * uy + c1
            b2 = vx + vy + c2
            b3 = torch.sqrt((vx * vy).clamp_min_(0.0)) + c3
            ssim = (
                a1 / b1.clamp_min_(1e-12)
            ) * (
                a2 / b2.clamp_min_(1e-12)
            ) * (
                a3 / b3.clamp_min_(1e-12)
            )
            gene_sum[g_start:g_end] += ssim.sum(dim=0)

    per_gene = gene_sum / float(y_true.shape[0])
    return per_gene, per_gene.mean()


def _infer_full(
    model: InterventionStableGenerator,
    slice_data: SliceData,
    device: str,
    batch_size: int,
    cfg: TrainConfig,
) -> dict[str, torch.Tensor]:
    model.eval()
    stable_list = []
    private_list = []
    gate_list = []
    view_weight_list = []
    local_spot_ids_full = torch.tensor(slice_data.spot_index, dtype=torch.long, device=device)
    spot_count_full = None
    if cfg.use_spot_count_feature:
        spot_count_full = torch.tensor(
            np.log1p(slice_data.spot_sizes[slice_data.spot_index]).astype(np.float32),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
    with torch.no_grad():
        for start in range(0, slice_data.he_cell.shape[0], batch_size):
            end = min(start + batch_size, slice_data.he_cell.shape[0])
            he = torch.tensor(slice_data.he_cell[start:end], dtype=torch.float32, device=device)
            spot_count = None if spot_count_full is None else spot_count_full[start:end]
            x_orig = _append_spot_count(he, spot_count, cfg)
            x_rand = x_orig
            x_comp1 = x_orig
            x_comp2 = x_orig
            outputs = model(
                x_orig=x_orig,
                x_rand=x_rand,
                x_comp1=x_comp1,
                x_comp2=x_comp2,
                local_spot_ids=local_spot_ids_full[start:end],
                spot_adj=None,
            )
            stable_list.append(outputs["stable"])
            private_list.append(outputs["private"])
            gate_list.append(outputs["gate"])
            view_weight_list.append(outputs["view_weights"])

    stable = torch.cat(stable_list, dim=0)
    private = torch.cat(private_list, dim=0)
    zero_111 = torch.zeros_like(private)
    gate = torch.cat(gate_list, dim=0)
    #view_weights = torch.cat(view_weight_list, dim=0)
    local_spot_ids = local_spot_ids_full
    spot_adj = _build_spot_adj(torch.tensor(slice_data.spot_spatial, dtype=torch.float32, device=device), cfg.gcn_knn) if cfg.use_resgcn else None
    stable = model.apply_spot_gcn(stable, local_spot_ids, spot_adj)
    pred_self = model.decode(stable, zero_111)

    num_spots = slice_data.spot_expr.shape[0]
    spot_self = torch.zeros((num_spots, pred_self.shape[1]), dtype=pred_self.dtype, device=device)
    spot_self.index_add_(0, local_spot_ids, pred_self)
    denom = torch.tensor(slice_data.spot_sizes, dtype=pred_self.dtype, device=device).clamp_min(1.0).unsqueeze(1)
    spot_self = spot_self / denom

    return {
        "pred_self": pred_self,
        "stable": stable,
        "private": private,
        "gate": gate,
        #"view_weights": view_weights,
        "spot_self": spot_self,
    }


def _evaluate_slice(slice_data: SliceData, outputs: dict[str, np.ndarray], cfg: TrainConfig) -> dict[str, float]:
    device = torch.device(cfg.device)
    y_cell_true = torch.tensor(slice_data.expr_cell, dtype=torch.float32, device=device)
    y_cell_pred = outputs["pred_self"].to(device=device, dtype=torch.float32)
    y_spot_true = torch.tensor(slice_data.spot_expr, dtype=torch.float32, device=device)
    y_spot_pred = outputs["spot_self"].to(device=device, dtype=torch.float32)
    spot_coords = torch.tensor(slice_data.spot_spatial, dtype=torch.float32, device=device)
    knn_idx = _build_spot_knn(spot_coords, cfg.eval_spot_knn)

    _, cell_pcc = _gpu_pcc(y_cell_true, y_cell_pred)
    _, cell_cmd = _gpu_cmd(y_cell_true, y_cell_pred)
    _, spot_pcc = _gpu_pcc(y_spot_true, y_spot_pred)
    _, spot_cmd = _gpu_cmd(y_spot_true, y_spot_pred)
    _, spot_ssim = _gpu_graph_ssim(y_spot_true, y_spot_pred, knn_idx)

    genes_to_plot = ["EPCAM", "ESR1", "PGR"]
    gene_scores = {}
    gene_pcc_all, _ = _gpu_pcc(y_cell_true, y_cell_pred)
    for gene_name in genes_to_plot:
        gene_idx = int(np.where(slice_data.var_names == gene_name)[0][0])
        gene_scores[f"gene_{gene_name.lower()}_pcc"] = float(gene_pcc_all[gene_idx].detach().cpu())

    cell_coords = torch.tensor(slice_data.spatial_cell, dtype=torch.float32, device=device)
    print(
        f"stage=cell_ssim_gpu knn={cfg.eval_cell_knn} sigma={cfg.eval_cell_ssim_sigma}",
        flush=True,
    )
    cell_knn_idx, cell_knn_w = _build_spatialex_style_knn_graph_gpu(
        cell_coords,
        num_neighbors=cfg.eval_cell_knn,
        sigma=cfg.eval_cell_ssim_sigma,
    )
    _, cell_ssim = _spatialex_style_ssim_gpu(y_cell_true, y_cell_pred, cell_knn_idx, cell_knn_w)
    gene_var = y_cell_true.var(dim=0, unbiased=False)
    topk = min(50, int(gene_var.shape[0]))
    top_idx = torch.topk(gene_var, k=topk, largest=True).indices
    top50_var_gene_pcc_mean = float(gene_pcc_all[top_idx].mean().detach().cpu())

    return {
        "cell_pcc": float(cell_pcc.detach().cpu()),
        "cell_cmd": float(cell_cmd.detach().cpu()),
        "cell_ssim": float(cell_ssim),
        "spot_pcc": float(spot_pcc.detach().cpu()),
        "spot_cmd": float(spot_cmd.detach().cpu()),
        "spot_ssim": float(spot_ssim.detach().cpu()),
        "top50_var_gene_pcc_mean": top50_var_gene_pcc_mean,
        **gene_scores,
    }


def _move_outputs_to_cpu(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in outputs.items()}


def run_experiment(cfg: TrainConfig) -> pd.DataFrame:
    _set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    print("stage=read_slice1", flush=True)
    slice1 = read_slice(
        cfg.adata1_path,
        "rep1",
        preprocess_device=cfg.device,
    )
    print("stage=read_slice2", flush=True)
    slice2 = read_slice(
        cfg.adata2_path,
        "rep2",
        preprocess_device=cfg.device,
    )

    if not np.array_equal(slice1.var_names, slice2.var_names):
        raise ValueError("The two slices do not share the same gene order.")

    print(
        f"stage=data_ready rep1_cells={slice1.expr_cell.shape[0]} rep1_spots={slice1.spot_expr.shape[0]} "
        f"rep2_cells={slice2.expr_cell.shape[0]} rep2_spots={slice2.spot_expr.shape[0]}",
        flush=True,
    )
    print("stage=move_train_data_to_gpu_cache", flush=True)
    slice1_cache = build_device_cache(slice1, cfg.device)

    print("stage=model_init", flush=True)
    model = InterventionStableGenerator(
        in_dim=slice1.he_cell.shape[1] + (1 if cfg.use_spot_count_feature else 0),
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        out_dim=slice1.expr_cell.shape[1],
        num_layers=cfg.num_layers,
        private_dropout=cfg.private_dropout,
        use_resgcn=cfg.use_resgcn,
        gcn_layers=cfg.gcn_layers,
        gcn_dropout=cfg.gcn_dropout,
        gcn_residual_scale=cfg.gcn_residual_scale,
        num_views=cfg.num_views,
        use_spot_count_feature=cfg.use_spot_count_feature,
        view_mixer_layers=cfg.view_mixer_layers,
        view_mixer_dropout=cfg.view_mixer_dropout,
    ).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    all_train_spot_ids = torch.arange(slice1.spot_expr.shape[0], device=cfg.device, dtype=torch.long)
    val_spot_ids = torch.empty(0, device=cfg.device, dtype=torch.long)
    if cfg.val_spot_frac > 0.0:
        val_count = max(1, int(round(slice1.spot_expr.shape[0] * cfg.val_spot_frac)))
        perm = torch.randperm(slice1.spot_expr.shape[0], device=cfg.device, dtype=torch.long)
        val_spot_ids = perm[:val_count]
        all_train_spot_ids = perm[val_count:]
        print(
            f"stage=spot_split train_spots={all_train_spot_ids.shape[0]} val_spots={val_spot_ids.shape[0]}",
            flush=True,
        )#验证集没啥意义，因为是跨切片，删了得了
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    for epoch in range(cfg.epochs):
        model.train()
        batches1 = _iter_spot_batches_from_ids(all_train_spot_ids, cfg.spot_batch_size)#在偏态分布与构图问题里取舍，我觉得还是保证分布相同把，不然训练每次都偏
        num_steps = len(batches1)
        epoch_stats = []
        print(f"stage=epoch_start epoch={epoch} steps={num_steps}", flush=True)

        for step in range(num_steps):
            batch1 = build_spot_batch_from_cache(slice1_cache, batches1[step])

            x1_orig, x1_rand, x1_comp1, x1_comp2 = _make_interventions(
                batch1["he"],
                cfg.rand_keep_prob,
                cfg.comp_focus_min_prob,
                cfg.comp_focus_max_prob,
                cfg.comp_focus_temperature,
                cfg.comp_focus_center,
            )
            x1_orig = _append_spot_count(x1_orig, batch1.get("spot_count"), cfg)
            x1_rand = _append_spot_count(x1_rand, batch1.get("spot_count"), cfg)
            x1_comp1 = _append_spot_count(x1_comp1, batch1.get("spot_count"), cfg)
            x1_comp2 = _append_spot_count(x1_comp2, batch1.get("spot_count"), cfg)

            out1 = model(
                x_orig=x1_orig,
                x_rand=x1_rand,
                x_comp1=x1_comp1,
                x_comp2=x1_comp2,
                local_spot_ids=batch1["local_spot_ids"],
                spot_adj=_build_spot_adj(batch1["spot_spatial"], cfg.gcn_knn) if cfg.use_resgcn else None,
            )

            loss1, stats1, _ = _slice_losses(out1, batch1["spot_expr"], batch1["local_spot_ids"], batch1["num_spots"], cfg)
            loss = loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_stats = {
                "loss": float(loss.detach().cpu()),
                "slice1_self": stats1["self"],
                "slice1_stable": stats1["stable"],
                "slice1_inv": stats1["inv"],
            }
            epoch_stats.append(step_stats)
            if step == 0 or (step + 1) % 8 == 0 or step + 1 == num_steps:
                print(
                    f"epoch={epoch} step={step + 1}/{num_steps} "
                    f"loss={step_stats['loss']:.4f} "
                    f"self={step_stats['slice1_self']:.4f} stable={step_stats['slice1_stable']:.4f} "
                    f"inv={step_stats['slice1_inv']:.4f} "
                    f"wself={cfg.self_weight * step_stats['slice1_self']:.4f} "
                    f"wstable={cfg.stable_weight * step_stats['slice1_stable']:.4f} "
                    f"winv={cfg.invariance_weight * step_stats['slice1_inv']:.4f}",
                    flush=True,
                )

        mean_loss = float(np.mean([item["loss"] for item in epoch_stats]))
        mean_self = float(np.mean([item["slice1_self"] for item in epoch_stats]))
        mean_stable = float(np.mean([item["slice1_stable"] for item in epoch_stats]))
        mean_inv = float(np.mean([item["slice1_inv"] for item in epoch_stats]))
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "slice1_self": mean_self,
                "slice1_stable": mean_stable,
                "slice1_inv": mean_inv,
            }
        )
        val_loss = float("nan")
        val_self = float("nan")
        if val_spot_ids.numel() > 0:
            val_loss, val_self = _evaluate_spot_loss(
                model,
                slice1_cache,
                val_spot_ids,
                cfg,
            )
            history[-1]["val_loss"] = val_loss
            history[-1]["val_self"] = val_self
        print(
            f"epoch={epoch} loss={mean_loss:.4f} "
            f"self={mean_self:.4f} stable={mean_stable:.4f} "
            f"inv={mean_inv:.4f} "
            f"wself={cfg.self_weight * mean_self:.4f} "
            f"wstable={cfg.stable_weight * mean_stable:.4f} "
            f"winv={cfg.invariance_weight * mean_inv:.4f}"
            + (
                f" val_loss={val_loss:.4f} val_self={val_self:.4f}"
                if val_spot_ids.numel() > 0
                else ""
            ),
            flush=True,
        )
        if val_spot_ids.numel() > 0:
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif cfg.early_stop_patience > 0 and epoch - best_epoch >= cfg.early_stop_patience:
                print(
                    f"stage=early_stop epoch={epoch} best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}",
                    flush=True,
                )
                break

    if best_state is not None:
        print(f"stage=restore_best_model best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}", flush=True)
        model.load_state_dict(best_state)

    print("stage=infer_train_slice1", flush=True)
    infer1 = _infer_full(
        model,
        slice1,
        device=cfg.device,
        batch_size=2048,
        cfg=cfg,
    )
    print("stage=infer_test_slice2", flush=True)
    infer2 = _infer_full(
        model,
        slice2,
        device=cfg.device,
        batch_size=2048,
        cfg=cfg,
    )
    infer1_cpu = _move_outputs_to_cpu(infer1)
    infer2_cpu = _move_outputs_to_cpu(infer2)
    del infer1
    del infer2
    del slice1_cache
    del optimizer
    if best_state is not None:
        del best_state
    model = model.cpu()
    torch.cuda.empty_cache()
    print("stage=evaluate", flush=True)
    eval1 = _evaluate_slice(slice1, infer1_cpu, cfg)
    eval2 = _evaluate_slice(slice2, infer2_cpu, cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"stage=save output_dir={output_dir}", flush=True)
    np.savez_compressed(
        output_dir / "embeddings.npz",
        stable1=infer1_cpu["stable"].numpy(),
        private1=infer1_cpu["private"].numpy(),
        gate1=infer1_cpu["gate"].numpy(),
        #view_weights1=infer1_cpu["view_weights"].numpy(),
        stable2=infer2_cpu["stable"].numpy(),
        private2=infer2_cpu["private"].numpy(),
        gate2=infer2_cpu["gate"].numpy(),
        #view_weights2=infer2_cpu["view_weights"].numpy(),
        spot_self1=infer1_cpu["spot_self"].numpy(),
        spot_self2=infer2_cpu["spot_self"].numpy(),
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(cfg),
        },
        output_dir / "model_state.pt",
    )

    summary = pd.DataFrame([{
        "device": cfg.device,
        "epochs": cfg.epochs,
        "num_views": cfg.num_views,
        "num_layers": cfg.num_layers,
        "hidden_dim": cfg.hidden_dim,
        "latent_dim": cfg.latent_dim,
        "spot_batch_size": cfg.spot_batch_size,
        "supervision": "spot",
        "model_kind": "intervention_stable_multiview_spot",
        "use_spot_count_feature": cfg.use_spot_count_feature,
        "use_resgcn": cfg.use_resgcn,
        "gcn_layers": cfg.gcn_layers,
        "gcn_knn": cfg.gcn_knn,
        "gcn_dropout": cfg.gcn_dropout,
        "gcn_residual_scale": cfg.gcn_residual_scale,
        "view_mixer_layers": cfg.view_mixer_layers,
        "view_mixer_dropout": cfg.view_mixer_dropout,
        "self_weight": cfg.self_weight,
        "stable_weight": cfg.stable_weight,
        "invariance_weight": cfg.invariance_weight,
        "val_spot_frac": cfg.val_spot_frac,
        "early_stop_patience": cfg.early_stop_patience,
        "private_dropout": cfg.private_dropout,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if np.isfinite(best_val_loss) else np.nan,
        "train_slice": slice1.name,
        "test_slice": slice2.name,
        "train_slice1_cell_pcc": eval1["cell_pcc"],
        "train_slice1_cell_cmd": eval1["cell_cmd"],
        "train_slice1_cell_ssim": eval1["cell_ssim"],
        "train_slice1_spot_pcc": eval1["spot_pcc"],
        "train_slice1_spot_ssim": eval1["spot_ssim"],
        "train_slice1_spot_cmd": eval1["spot_cmd"],
        "train_top50_var_gene_pcc_mean": eval1["top50_var_gene_pcc_mean"],
        "test_slice2_cell_pcc": eval2["cell_pcc"],
        "test_slice2_cell_cmd": eval2["cell_cmd"],
        "test_slice2_cell_ssim": eval2["cell_ssim"],
        "test_slice2_spot_pcc": eval2["spot_pcc"],
        "test_slice2_spot_ssim": eval2["spot_ssim"],
        "test_slice2_spot_cmd": eval2["spot_cmd"],
        "test_top50_var_gene_pcc_mean": eval2["top50_var_gene_pcc_mean"],
        "test_gene_epcam_pcc": eval2["gene_epcam_pcc"],
        "test_gene_esr1_pcc": eval2["gene_esr1_pcc"],
        "test_gene_pgr_pcc": eval2["gene_pgr_pcc"],
        "slice1_num_cells": slice1.expr_cell.shape[0],
        "slice2_num_cells": slice2.expr_cell.shape[0],
        "slice1_num_spots": slice1.spot_expr.shape[0],
        "slice2_num_spots": slice2.spot_expr.shape[0],
    }])
    result_csv = Path(cfg.result_csv)
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    if result_csv.exists():
        existing = pd.read_csv(result_csv)
        summary = pd.concat([existing, summary], ignore_index=True)
    summary.to_csv(result_csv, index=False)
    return summary
