from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scanpy as sc
import torch


@dataclass
class SliceData:
    name: str
    expr_cell: np.ndarray
    he_cell: np.ndarray
    spatial_cell: np.ndarray
    obs_names: np.ndarray
    var_names: np.ndarray
    raw_spot_ids: np.ndarray
    spot_index: np.ndarray
    spot_expr: np.ndarray
    spot_he: np.ndarray
    spot_spatial: np.ndarray
    spot_sizes: np.ndarray
    spot_to_cells: list[np.ndarray]
    spot_cell_ids_flat: np.ndarray
    spot_local_ids_flat: np.ndarray
    spot_offsets: np.ndarray

def _to_dense(x):
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float32)
    return np.asarray(x.toarray(), dtype=np.float32)


def _torch_device(device: str | None) -> torch.device | None:
    if device is None:
        return None
    return torch.device(device)


def _normalize_he(he: np.ndarray) -> np.ndarray:
    he_min = he.min(axis=0, keepdims=True)
    he_max = he.max(axis=0, keepdims=True)
    he_range = np.clip(he_max - he_min, a_min=1e-6, a_max=None)
    return np.clip((he - he_min) / he_range, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_he_gpu(he: np.ndarray, device: str) -> np.ndarray:
    he_t = torch.as_tensor(he, dtype=torch.float32, device=device)
    he_min = he_t.amin(dim=0, keepdim=True)
    he_max = he_t.amax(dim=0, keepdim=True)
    he_range = (he_max - he_min).clamp_min(1e-6)
    he_t = ((he_t - he_min) / he_range).clamp_(0.0, 1.0)
    return he_t.cpu().numpy().astype(np.float32, copy=False)


def _group_by_spot(values: np.ndarray, spot_index: np.ndarray, num_spots: int) -> np.ndarray:
    out = np.zeros((num_spots, values.shape[1]), dtype=np.float32)
    np.add.at(out, spot_index, values)
    counts = np.bincount(spot_index, minlength=num_spots).astype(np.float32)
    out /= np.clip(counts[:, None], a_min=1.0, a_max=None)
    return out


def _group_by_spot_gpu(values: np.ndarray, spot_index: np.ndarray, num_spots: int, device: str) -> np.ndarray:
    values_t = torch.as_tensor(values, dtype=torch.float32, device=device)
    spot_index_t = torch.as_tensor(spot_index, dtype=torch.long, device=device)
    out = torch.zeros((num_spots, values_t.shape[1]), dtype=torch.float32, device=device)
    out.index_add_(0, spot_index_t, values_t)
    counts = torch.bincount(spot_index_t, minlength=num_spots).to(dtype=torch.float32)
    out = out / counts.clamp_min(1.0).unsqueeze(1)
    return out.cpu().numpy().astype(np.float32, copy=False)


def read_slice(
    path: str,
    name: str,
    preprocess_device: str | None = None,
) -> SliceData:
    print(f"stage=read_slice_h5ad_start slice={name} path={path}", flush=True)
    adata = sc.read_h5ad(path)
    print(f"stage=read_slice_h5ad_done slice={name} cells={adata.n_obs} genes={adata.n_vars}", flush=True)
    print(f"stage=read_slice_dense_x_start slice={name}", flush=True)
    expr_cell = _to_dense(adata.X)
    print(f"stage=read_slice_dense_x_done slice={name} shape={expr_cell.shape}", flush=True)
    use_gpu_preprocess = preprocess_device is not None and _torch_device(preprocess_device).type == "cuda"
    print(f"stage=read_slice_he_start slice={name}", flush=True)
    he_array = np.asarray(adata.obsm["he"], dtype=np.float32)
    he_cell = _normalize_he_gpu(he_array, preprocess_device) if use_gpu_preprocess else _normalize_he(he_array)
    print(f"stage=read_slice_he_done slice={name} shape={he_cell.shape}", flush=True)
    print(f"stage=read_slice_spatial_start slice={name}", flush=True)
    spatial_cell = np.asarray(adata.obsm["spatial"][:, :2], dtype=np.float32)
    print(f"stage=read_slice_spatial_done slice={name} shape={spatial_cell.shape}", flush=True)
    print(f"stage=read_slice_obsvar_start slice={name}", flush=True)
    obs_names = np.asarray(adata.obs_names.astype(str), dtype=object)
    var_names = np.asarray(adata.var_names.astype(str), dtype=object)
    raw_spot_ids = np.asarray(adata.obs["spot"], dtype=np.int64)
    print(f"stage=read_slice_obsvar_done slice={name} obs={obs_names.shape[0]} vars={var_names.shape[0]}", flush=True)
    print(f"stage=read_slice_unique_spot_start slice={name}", flush=True)
    uniq_spots, spot_index = np.unique(raw_spot_ids, return_inverse=True)
    num_spots = uniq_spots.shape[0]
    print(f"stage=read_slice_unique_spot_done slice={name} spots={num_spots}", flush=True)

    print(f"stage=read_slice_group_expr_start slice={name}", flush=True)
    spot_expr = (
        _group_by_spot_gpu(expr_cell, spot_index, num_spots, preprocess_device)
        if use_gpu_preprocess
        else _group_by_spot(expr_cell, spot_index, num_spots)
    )
    print(f"stage=read_slice_group_expr_done slice={name} shape={spot_expr.shape}", flush=True)
    print(f"stage=read_slice_group_he_start slice={name}", flush=True)
    spot_he = (
        _group_by_spot_gpu(he_cell, spot_index, num_spots, preprocess_device)
        if use_gpu_preprocess
        else _group_by_spot(he_cell, spot_index, num_spots)
    )
    print(f"stage=read_slice_group_he_done slice={name} shape={spot_he.shape}", flush=True)
    print(f"stage=read_slice_group_spatial_start slice={name}", flush=True)
    spot_spatial = (
        _group_by_spot_gpu(spatial_cell, spot_index, num_spots, preprocess_device)
        if use_gpu_preprocess
        else _group_by_spot(spatial_cell, spot_index, num_spots)
    )
    print(f"stage=read_slice_group_spatial_done slice={name} shape={spot_spatial.shape}", flush=True)
    print(f"stage=read_slice_spot_sizes_start slice={name}", flush=True)
    spot_sizes = np.bincount(spot_index, minlength=num_spots).astype(np.int64)
    print(f"stage=read_slice_spot_sizes_done slice={name}", flush=True)
    print(f"stage=read_slice_spot_to_cells_start slice={name}", flush=True)
    spot_to_cells = [np.flatnonzero(spot_index == i).astype(np.int64) for i in range(num_spots)]
    print(f"stage=read_slice_spot_to_cells_done slice={name}", flush=True)
    print(f"stage=read_slice_offsets_start slice={name}", flush=True)
    spot_offsets = np.zeros(num_spots + 1, dtype=np.int64)
    spot_offsets[1:] = np.cumsum(spot_sizes)
    print(f"stage=read_slice_offsets_done slice={name}", flush=True)
    print(f"stage=read_slice_flat_index_start slice={name}", flush=True)
    spot_cell_ids_flat = np.empty(spot_index.shape[0], dtype=np.int64)
    spot_local_ids_flat = np.empty(spot_index.shape[0], dtype=np.int64)
    cursor = 0
    for spot_id, members in enumerate(spot_to_cells):
        next_cursor = cursor + members.shape[0]
        spot_cell_ids_flat[cursor:next_cursor] = members
        spot_local_ids_flat[cursor:next_cursor] = spot_id
        cursor = next_cursor
    print(f"stage=read_slice_flat_index_done slice={name}", flush=True)

    print(f"stage=read_slice_pack_start slice={name}", flush=True)
    slice_data = SliceData(
        name=name,
        expr_cell=expr_cell,
        he_cell=he_cell,
        spatial_cell=spatial_cell,
        obs_names=obs_names,
        var_names=var_names,
        raw_spot_ids=uniq_spots.astype(np.int64, copy=False),
        spot_index=spot_index.astype(np.int64, copy=False),
        spot_expr=spot_expr,
        spot_he=spot_he,
        spot_spatial=spot_spatial,
        spot_sizes=spot_sizes,
        spot_to_cells=spot_to_cells,
        spot_cell_ids_flat=spot_cell_ids_flat,
        spot_local_ids_flat=spot_local_ids_flat,
        spot_offsets=spot_offsets,
    )
    print(f"stage=read_slice_pack_done slice={name}", flush=True)
    print(f"stage=read_slice_return slice={name}", flush=True)
    return slice_data


def build_spot_batch(slice_data: SliceData, spot_ids: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    counts = slice_data.spot_sizes[spot_ids]
    total_cells = int(counts.sum())
    cell_ids = np.empty(total_cells, dtype=np.int64)
    local_spot_ids = np.empty(total_cells, dtype=np.int64)
    cursor = 0
    for local_id, spot_id in enumerate(spot_ids.tolist()):
        start = int(slice_data.spot_offsets[spot_id])
        end = int(slice_data.spot_offsets[spot_id + 1])
        block = slice_data.spot_cell_ids_flat[start:end]
        next_cursor = cursor + block.shape[0]
        cell_ids[cursor:next_cursor] = block
        local_spot_ids[cursor:next_cursor] = local_id
        cursor = next_cursor
    return {
        "cell_ids": torch.tensor(cell_ids, dtype=torch.long, device=device),
        "local_spot_ids": torch.tensor(local_spot_ids, dtype=torch.long, device=device),
        "he": torch.tensor(slice_data.he_cell[cell_ids], dtype=torch.float32, device=device),
        "spot_count": torch.tensor(np.log1p(counts[local_spot_ids]).astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1),
        "spot_expr": torch.tensor(slice_data.spot_expr[spot_ids], dtype=torch.float32, device=device),
        "spot_spatial": torch.tensor(slice_data.spot_spatial[spot_ids], dtype=torch.float32, device=device),
        "domain": torch.zeros(cell_ids.shape[0], dtype=torch.long, device=device),
        "num_spots": int(spot_ids.shape[0]),
    }


def build_device_cache(slice_data: SliceData, device: str) -> dict[str, torch.Tensor]:
    return {
        "he_cell": torch.tensor(slice_data.he_cell, dtype=torch.float32, device=device),
        "spot_expr": torch.tensor(slice_data.spot_expr, dtype=torch.float32, device=device),
        "spot_spatial": torch.tensor(slice_data.spot_spatial, dtype=torch.float32, device=device),
        "spot_sizes": torch.tensor(slice_data.spot_sizes, dtype=torch.long, device=device),
        "spot_sizes_log1p": torch.tensor(np.log1p(slice_data.spot_sizes).astype(np.float32), dtype=torch.float32, device=device),
        "spot_offsets": torch.tensor(slice_data.spot_offsets, dtype=torch.long, device=device),
        "spot_cell_ids_flat": torch.tensor(slice_data.spot_cell_ids_flat, dtype=torch.long, device=device),
    }


def build_spot_batch_from_cache(cache: dict[str, torch.Tensor], spot_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    device = spot_ids.device
    counts = cache["spot_sizes"].index_select(0, spot_ids)
    max_count = int(counts.max().item())
    base = torch.arange(max_count, device=device, dtype=torch.long).unsqueeze(0)
    starts = cache["spot_offsets"].index_select(0, spot_ids).unsqueeze(1)
    mask = base < counts.unsqueeze(1)
    gather_pos = (starts + base)[mask]
    cell_ids = cache["spot_cell_ids_flat"].index_select(0, gather_pos)
    local_spot_ids = torch.repeat_interleave(
        torch.arange(spot_ids.shape[0], device=device, dtype=torch.long),
        counts,
    )
    return {
        "cell_ids": cell_ids,
        "local_spot_ids": local_spot_ids,
        "he": cache["he_cell"].index_select(0, cell_ids),
        "spot_count": cache["spot_sizes_log1p"].index_select(0, spot_ids).index_select(0, local_spot_ids).unsqueeze(1),
        "spot_expr": cache["spot_expr"].index_select(0, spot_ids),
        "spot_spatial": cache["spot_spatial"].index_select(0, spot_ids),
        "domain": torch.zeros(cell_ids.shape[0], dtype=torch.long, device=device),
        "num_spots": int(spot_ids.shape[0]),
    }
