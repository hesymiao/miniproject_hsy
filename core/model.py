import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        blocks = [MlpBlock(in_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            blocks.append(MlpBlock(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        hidden = x
        for block in self.blocks:
            hidden = block(hidden)
            outputs.append(hidden)
        return outputs


class ResidualGraphConv(nn.Module):
    def __init__(self, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(latent_dim, latent_dim, bias=False)
        self.norm = nn.LayerNorm(latent_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        msg = adj @ self.lin(x)
        delta = self.drop(self.act(self.norm(msg)))
        return x + delta


class ViewInteractionBlock(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(token_dim, token_dim, bias=False)
        self.k = nn.Linear(token_dim, token_dim, bias=False)
        self.v = nn.Linear(token_dim, token_dim, bias=False)
        self.mix = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_dim),
        )
        self.norm = nn.LayerNorm(token_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scale = tokens.shape[-1] ** -0.5
        attn = torch.softmax((self.q(tokens) @ self.k(tokens).transpose(-1, -2)) * scale, dim=-1) #标准attention
        mixed = attn @ self.v(tokens)
        # delta = self.mix(torch.cat([tokens, mixed], dim=-1)) 
        # return self.norm(tokens + self.drop(delta))
        return self.norm(tokens+mixed)


class ViewMixer(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, num_views: int, num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_views = num_views
        self.blocks = nn.ModuleList(
            [ViewInteractionBlock(token_dim, hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        pair_dim = token_dim * (num_views * (num_views - 1) // 2)
        summary_dim = token_dim * num_views + token_dim * 2 + pair_dim
        context_dim = token_dim * 2 + pair_dim
        self.weight_head = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_views),
        )
        self.context_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )
        # self.mlp_hebing = nn.Sequential(
        #     nn.Linear(3*token_dim, token_dim),
        #     nn.SiLU())
        self.out_norm = nn.LayerNorm(token_dim)

    def _pairwise_abs(self, tokens: torch.Tensor) -> torch.Tensor:
        pieces = []
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                pieces.append(torch.abs(tokens[:, i] - tokens[:, j]))
        if not pieces:
            return tokens.new_zeros((tokens.shape[0], 0))
        return torch.cat(pieces, dim=1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden)
        mean_token = hidden.mean(dim=1)
        max_token = hidden.amax(dim=1)
        pairwise = self._pairwise_abs(hidden)
        summary = torch.cat([hidden.flatten(start_dim=1), mean_token, max_token, pairwise], dim=1)
        view_weights = torch.softmax(self.weight_head(summary), dim=1)
        fused = torch.sum(hidden * view_weights.unsqueeze(-1), dim=1)
        fused = self.out_norm(fused + self.context_head(torch.cat([mean_token, max_token, pairwise], dim=1)))
        return hidden, fused, view_weights
        # fused = self.mlp_hebing(hidden.flatten(start_dim=1))
        # fused = self.out_norm(fused)
        # return hidden, fused,0


class InterventionStableGenerator(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        out_dim: int,
        num_layers: int = 3,
        private_dropout: float = 0.3,
        use_resgcn: bool = False,
        gcn_layers: int = 2,
        gcn_dropout: float = 0.1,
        gcn_residual_scale: float = 0.2,
        num_views: int = 3,
        use_spot_count_feature: bool = False,
        view_mixer_layers: int = 2,
        view_mixer_dropout: float = 0.1,
    ):
        super().__init__()
        self.private_dropout = private_dropout
        self.use_resgcn = use_resgcn
        self.gcn_residual_scale = gcn_residual_scale
        self.num_views = num_views
        self.count_dim = 1 if use_spot_count_feature else 0
        image_dim = in_dim - self.count_dim
        if image_dim % num_views != 0:
            raise ValueError(f"input dim {image_dim} cannot be split into {num_views} equal views")
        self.view_dims = [image_dim // num_views for _ in range(num_views)]
        self.view_stems = nn.ModuleList([MlpBlock(view_dim, hidden_dim) for view_dim in self.view_dims])
        self.encoder = MultiScaleEncoder(hidden_dim + self.count_dim, hidden_dim, num_layers)
        self.joined_dim = hidden_dim * num_layers
        self.view_token_proj = nn.Sequential(
            nn.Linear(self.joined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.view_mixer = ViewMixer(
            latent_dim,
            hidden_dim,
            num_views,
            num_blocks=view_mixer_layers,
            dropout=view_mixer_dropout,
        )
        shared_input_dim = self.joined_dim * num_views + latent_dim
        self.shared_proj = nn.Sequential(
            nn.Linear(shared_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.private_proj = nn.Sequential(
            nn.Linear(self.joined_dim * num_views, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.spot_gcn = nn.ModuleList(
            [ResidualGraphConv(latent_dim, dropout=gcn_dropout) for _ in range(gcn_layers)]
        )
        self.stable_norm = nn.LayerNorm(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _split_views(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        if self.count_dim > 0:
            count_x = x[:, -self.count_dim :]
            x = x[:, :-self.count_dim]
        else:
            count_x = None
        views = list(x.split(self.view_dims, dim=1))
        return views, count_x

    def _encode_one(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #每个 view 先各自编码，再 token 化，再做 view mixer
        views, count_x = self._split_views(x)
        view_levels = []
        view_joined = []
        for view_idx, view in enumerate(views):
            hidden = self.view_stems[view_idx](view)#mlp(view)
            if count_x is not None:
                hidden = torch.cat([hidden, count_x], dim=1)
            levels = self.encoder(hidden) #num_layers,batch,hidden
            view_levels.append(levels)
            view_joined.append(torch.cat(levels, dim=1))#[batch, hidden_dim * num_layers]
        joined_stack = torch.stack(view_joined, dim=1)#[batch, num_views, hidden_dim * num_layers]
        view_tokens = self.view_token_proj(joined_stack) #[batch, num_views, latent_dim]
        refined_tokens, fused_context, view_weights = self.view_mixer(view_tokens)
        shared_input = torch.cat([joined_stack.flatten(start_dim=1), fused_context], dim=1)
        return view_levels, shared_input, joined_stack.flatten(start_dim=1), refined_tokens, view_weights

    def _make_gate(
        self,
        shared_orig: torch.Tensor,
        shared_rand: torch.Tensor,
        shared_comp1: torch.Tensor,
        shared_comp2: torch.Tensor,
    ) -> torch.Tensor:
        gate_input = torch.cat(
            [
                shared_orig,
                shared_rand,
                0.5 * (shared_comp1 + shared_comp2),
                torch.abs(shared_orig - shared_rand),
                torch.abs(shared_comp1 - shared_comp2),
            ],
            dim=1,
        )
        return self.gate(gate_input)

    def encode(
        self,
        x_orig: torch.Tensor,
        x_rand: torch.Tensor,
        x_comp1: torch.Tensor,
        x_comp2: torch.Tensor,
        local_spot_ids: torch.Tensor | None = None,
        spot_adj: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        orig_levels, orig_shared_input, orig_joined, orig_refined_tokens, orig_view_weights = self._encode_one(x_orig)
        _, rand_shared_input, rand_joined, rand_refined_tokens, rand_view_weights = self._encode_one(x_rand)
        _, comp1_shared_input, comp1_joined, comp1_refined_tokens, comp1_view_weights = self._encode_one(x_comp1)
        _, comp2_shared_input, comp2_joined, comp2_refined_tokens, comp2_view_weights = self._encode_one(x_comp2)

        shared_orig = self.shared_proj(orig_shared_input)
        shared_rand = self.shared_proj(rand_shared_input)
        shared_comp1 = self.shared_proj(comp1_shared_input)
        shared_comp2 = self.shared_proj(comp2_shared_input)
        gate = self._make_gate(shared_orig, shared_rand, shared_comp1, shared_comp2)
        intervention_mean = (shared_rand + shared_comp1 + shared_comp2) / 3.0
        stable_base = gate * shared_orig + (1.0 - gate) * intervention_mean

        stable = self.stable_norm(stable_base)
        if self.use_resgcn and local_spot_ids is not None and spot_adj is not None and spot_adj.shape[0] > 1:
            stable = self.apply_spot_gcn(stable, local_spot_ids, spot_adj)
        private = self.private_proj(orig_joined)

        return {
            "levels": orig_levels,
            "shared_orig": shared_orig,
            "shared_rand": shared_rand,
            "shared_comp1": shared_comp1,
            "shared_comp2": shared_comp2,
            "stable": stable,
            "private": private,
            "gate": gate,
            "view_weights": orig_view_weights,
            "rand_view_weights": rand_view_weights,
            "comp1_view_weights": comp1_view_weights,
            "comp2_view_weights": comp2_view_weights,
            "refined_tokens": orig_refined_tokens,
            "rand_refined_tokens": rand_refined_tokens,
            "comp1_refined_tokens": comp1_refined_tokens,
            "comp2_refined_tokens": comp2_refined_tokens,
        }

    def decode(
        self,
        stable: torch.Tensor,
        private: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.decoder(torch.cat([stable, private], dim=1))
        return F.leaky_relu(pred)

    def apply_spot_gcn(
        self,
        stable: torch.Tensor,
        local_spot_ids: torch.Tensor,
        spot_adj: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_resgcn or spot_adj is None or spot_adj.shape[0] <= 1:
            return stable
        num_spots = spot_adj.shape[0]
        spot_stable = torch.zeros((num_spots, stable.shape[1]), dtype=stable.dtype, device=stable.device)
        counts = torch.zeros(num_spots, dtype=stable.dtype, device=stable.device)
        spot_stable.index_add_(0, local_spot_ids, stable)
        counts.index_add_(0, local_spot_ids, torch.ones_like(local_spot_ids, dtype=stable.dtype))
        spot_stable = spot_stable / counts.clamp_min(1.0).unsqueeze(1)
        spot_stable_gcn = spot_stable
        for layer in self.spot_gcn:
            spot_stable_gcn = layer(spot_stable_gcn, spot_adj)
        spot_delta = spot_stable_gcn.index_select(0, local_spot_ids) - stable
        return self.stable_norm(stable + self.gcn_residual_scale * spot_delta)

    def forward(
        self,
        x_orig: torch.Tensor,
        x_rand: torch.Tensor,
        x_comp1: torch.Tensor,
        x_comp2: torch.Tensor,
        local_spot_ids: torch.Tensor | None = None,
        spot_adj: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode(
            x_orig,
            x_rand,
            x_comp1,
            x_comp2,
            local_spot_ids=local_spot_ids,
            spot_adj=spot_adj,
        )
        if self.training and self.private_dropout > 0.0:
            private_self = F.dropout(enc["private"], p=self.private_dropout, training=True)
        else:
            private_self = enc["private"]
        private_zero = torch.zeros_like(enc["private"])
        pred_self = self.decode(enc["stable"], private_self)
        pred_stable = self.decode(enc["stable"], private_zero)
        enc["pred_self"] = pred_self
        enc["pred_stable"] = pred_stable
        return enc
