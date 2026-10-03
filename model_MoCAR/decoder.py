import math
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph

from model_MoCAR.chunk import VAEEncoder, VAEDecoder
from model_MoCAR.base import DecoderOnlyTrajectoryModel
from layers.fourier_embedding import FourierEmbedding
from layers.attention_layer import AttentionLayer
from layers.mlp_layer import MLPLayer
from utils import angle_between_2d_vectors, wrap_angle, weight_init


class ChunkAgent(DecoderOnlyTrajectoryModel):
    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        num_modes: int,
        num_future_chunks: Optional[int] = None,
        chunk_points: int = 11,
        min_valid_points: int = 5,
        chunk_ckpt_path: Optional[str] = None,
        use_aux_loss: bool = False,
        chunk_noise_ratio: float = 0.0,
        chunk_noise_sigma: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_modes = num_modes

        self.chunk_points = chunk_points
        self.chunk_stride = chunk_points - 1
        self.min_valid_points = min_valid_points
        self.use_aux_loss = use_aux_loss
        self.chunk_noise_ratio = chunk_noise_ratio
        self.chunk_noise_sigma = chunk_noise_sigma

        self.num_aux_decode_layers = min(2, max(0, num_layers - 1)) if use_aux_loss else 0
        self.aux_layer_ids = list(
            range(num_layers - 1 - self.num_aux_decode_layers, num_layers - 1)
        )

        assert self.chunk_points >= self.min_valid_points >= 2

        if num_future_chunks is None:
            self.num_future_chunks = math.ceil(num_future_steps / self.chunk_stride)
        else:
            self.num_future_chunks = num_future_chunks

        chunk_anchor = self._build_chunk_anchor(
            num_historical_steps=num_historical_steps,
            chunk_stride=self.chunk_stride,
        )
        self.register_buffer("chunk_anchor", chunk_anchor, persistent=False)
        self.num_chunks = int(chunk_anchor.numel())

        self.hist_emb = nn.Parameter(torch.randn(hidden_dim))
        self.mode_emb = nn.Embedding(num_modes, hidden_dim)

        self.chunk_encoder = VAEEncoder(
            input_dim=8,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            num_freq_bands=num_freq_bands,
            num_gru_layers=1,
        )
        self.chunk_proj = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
        )

        self.chunk_proj_back = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, z_dim),
        )

        self.chunk_decoder = VAEDecoder(
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            chunk_points=self.chunk_points,
        )

        self.r_t_emb = FourierEmbedding(
            input_dim=4,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pl2c_emb = FourierEmbedding(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_c2c_emb = FourierEmbedding(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )

        self.t_attn_layers = nn.ModuleList([
            AttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                bipartite=False,
                has_pos_emb=True,
            )
            for _ in range(num_layers)
        ])

        self.pl2c_attn_layers = nn.ModuleList([
            AttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                bipartite=True,
                has_pos_emb=True,
            )
            for _ in range(num_layers)
        ])

        self.c2c_attn_layers = nn.ModuleList([
            AttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                bipartite=False,
                has_pos_emb=True,
            )
            for _ in range(num_layers)
        ])

        self.m2m_attn_layer = AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=False,
            has_pos_emb=False,
        )
            

        self.prob_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)

        self.apply(weight_init)
        nn.init.zeros_(self.chunk_proj_back[-1].weight)
        nn.init.zeros_(self.chunk_proj_back[-1].bias)
        nn.init.normal_(self.hist_emb, mean=0.0, std=0.02)

        if chunk_ckpt_path is not None:
            self.load_pretrained_chunk(chunk_ckpt_path)

    def load_pretrained_chunk(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        encoder_state = {}
        decoder_state = {}

        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v
            elif k.startswith("decoder."):
                decoder_state[k[len("decoder."):]] = v

        if len(encoder_state) == 0:
            raise RuntimeError(f"No encoder weights found in checkpoint: {ckpt_path}")
        if len(decoder_state) == 0:
            raise RuntimeError(f"No decoder weights found in checkpoint: {ckpt_path}")

        self.chunk_encoder.load_state_dict(encoder_state, strict=True)
        self.chunk_decoder.load_state_dict(decoder_state, strict=True)
        # for p in self.chunk_encoder.parameters():
        #     p.requires_grad_(False)
        # for p in self.chunk_decoder.parameters():
        #     p.requires_grad_(False)

    def build_mode_spatial_inputs(
        self,
        pos_mode: torch.Tensor,
        head_mode: torch.Tensor,
        mode_valid: torch.Tensor,
        batch_agent_base: torch.Tensor,
        batch_pl_base: torch.Tensor,
        num_graphs: int,
        pos_pl: torch.Tensor,
        orient_pl: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor
    ]:
        device = pos_mode.device
        num_nodes = pos_mode.size(0)

        pos_mode_flat = pos_mode.reshape(-1, self.input_dim)
        head_mode_flat = head_mode.reshape(-1)
        head_vector_mode_flat = torch.stack(
            [head_mode_flat.cos(), head_mode_flat.sin()], dim=-1
        )
        mode_valid_flat = mode_valid.reshape(-1)

        batch_mode_pl2m = batch_agent_base.repeat_interleave(self.num_modes)
        edge_index_pl2m = radius(
            x=pos_mode_flat[:, :2], y=pos_pl[:, :2],
            r=self.pl2a_radius, batch_x=batch_mode_pl2m, batch_y=batch_pl_base,
            max_num_neighbors=300,
        )
        edge_index_pl2m = edge_index_pl2m[:, mode_valid_flat[edge_index_pl2m[1]]]

        if edge_index_pl2m.size(1) > 0:
            rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_mode_flat[edge_index_pl2m[1]]
            rel_orient_pl2m = wrap_angle(
                orient_pl[edge_index_pl2m[0]] - head_mode_flat[edge_index_pl2m[1]]
            )
            r_pl2m = torch.stack([
                torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_mode_flat[edge_index_pl2m[1]],
                    nbr_vector=rel_pos_pl2m[:, :2],
                ),
                rel_orient_pl2m,
            ], dim=-1)
            r_pl2m = self.r_pl2c_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        else:
            r_pl2m = torch.empty(0, self.hidden_dim, device=device, dtype=dtype)

        mode_slot = torch.arange(self.num_modes, device=device).repeat(num_nodes)
        batch_mode_c2c = batch_agent_base.repeat_interleave(self.num_modes) + mode_slot * num_graphs

        edge_index_mode_c2c = radius_graph(
            x=pos_mode_flat[:, :2], r=self.a2a_radius,
            batch=batch_mode_c2c, loop=False, max_num_neighbors=300,
        )
        edge_index_mode_c2c = subgraph(subset=mode_valid_flat, edge_index=edge_index_mode_c2c)[0]

        if edge_index_mode_c2c.size(1) > 0:
            rel_pos_mode_c2c = pos_mode_flat[edge_index_mode_c2c[0]] - pos_mode_flat[edge_index_mode_c2c[1]]
            rel_head_mode_c2c = wrap_angle(
                head_mode_flat[edge_index_mode_c2c[0]] - head_mode_flat[edge_index_mode_c2c[1]]
            )
            r_mode_c2c = torch.stack([
                torch.norm(rel_pos_mode_c2c[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_mode_flat[edge_index_mode_c2c[1]],
                    nbr_vector=rel_pos_mode_c2c[:, :2],
                ),
                rel_head_mode_c2c,
            ], dim=-1)
            r_mode_c2c = self.r_c2c_emb(continuous_inputs=r_mode_c2c, categorical_embs=None)
        else:
            r_mode_c2c = torch.empty(0, self.hidden_dim, device=device, dtype=dtype)

        edge_index_mode_m2m = self.build_same_agent_dense_edges(
            valid=mode_valid, loop=True
        )

        return (
            pos_mode_flat, head_mode_flat, head_vector_mode_flat, mode_valid_flat,
            edge_index_pl2m, r_pl2m,
            edge_index_mode_c2c, r_mode_c2c,
            edge_index_mode_m2m,
        )

    def build_temporal_geometry_to_mode_inputs(
        self,
        pos_hist: torch.Tensor,
        head_hist: torch.Tensor,
        valid_hist: torch.Tensor,
        anchor_hist: torch.Tensor,
        fut_pos_list: List[torch.Tensor],
        fut_head_list: List[torch.Tensor],
        fut_valid_list: List[torch.Tensor],
        fut_anchor_list: List[int],
        pos_mode: torch.Tensor,
        head_mode: torch.Tensor,
        mode_valid: torch.Tensor,
        current_anchor: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = pos_mode.device
        num_nodes = pos_mode.size(0)
        num_hist_chunks = valid_hist.size(1)

        pos_mode_flat = pos_mode.reshape(-1, self.input_dim)
        head_mode_flat = head_mode.reshape(-1)
        head_vector_mode_flat = torch.stack(
            [head_mode_flat.cos(), head_mode_flat.sin()], dim=-1
        )

        anchor_mode = torch.full(
            (num_nodes * self.num_modes,),
            fill_value=int(current_anchor),
            dtype=anchor_hist.dtype, device=device,
        )

        pos_hist_mem = pos_hist.reshape(num_nodes * num_hist_chunks, self.input_dim)
        head_hist_mem = head_hist.reshape(num_nodes * num_hist_chunks)
        anchor_hist_mem = anchor_hist[None, :].expand(num_nodes, -1).reshape(-1)

        edge_hist2mode = self.build_same_agent_bipartite_edges(
            src_valid=valid_hist, dst_valid=mode_valid,
        )

        if edge_hist2mode.size(1) > 0:
            rel_pos_hist = pos_hist_mem[edge_hist2mode[0]] - pos_mode_flat[edge_hist2mode[1]]
            rel_head_hist = wrap_angle(
                head_hist_mem[edge_hist2mode[0]] - head_mode_flat[edge_hist2mode[1]]
            )
            rel_dt_hist = anchor_hist_mem[edge_hist2mode[0]] - anchor_mode[edge_hist2mode[1]]
            r_hist2mode = torch.stack([
                torch.norm(rel_pos_hist[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_mode_flat[edge_hist2mode[1]],
                    nbr_vector=rel_pos_hist[:, :2],
                ),
                rel_head_hist,
                rel_dt_hist.float(),
            ], dim=-1)
            r_hist2mode = self.r_t_emb(continuous_inputs=r_hist2mode, categorical_embs=None)
        else:
            r_hist2mode = torch.empty(0, self.hidden_dim, device=device, dtype=dtype)

        edge_parts = [edge_hist2mode]
        r_parts = [r_hist2mode]

        if len(fut_pos_list) > 0:
            num_fut_chunks = len(fut_pos_list)
            pos_fut_mem = torch.stack(fut_pos_list, dim=2).contiguous()
            head_fut_mem = torch.stack(fut_head_list, dim=2).contiguous()
            valid_fut_mem = torch.stack(fut_valid_list, dim=2).contiguous().bool()
            anchor_fut = torch.tensor(fut_anchor_list, dtype=anchor_hist.dtype, device=device)

            pos_fut_flat = pos_fut_mem.reshape(num_nodes * self.num_modes * num_fut_chunks, self.input_dim)
            head_fut_flat = head_fut_mem.reshape(num_nodes * self.num_modes * num_fut_chunks)
            anchor_fut_flat = anchor_fut.view(1, 1, num_fut_chunks).expand(
                num_nodes, self.num_modes, -1
            ).reshape(-1)

            edge_fut2mode = self.build_same_agent_mode_bipartite_edges(
                src_valid=valid_fut_mem, dst_valid=mode_valid,
            )

            if edge_fut2mode.size(1) > 0:
                rel_pos_fut = pos_fut_flat[edge_fut2mode[0]] - pos_mode_flat[edge_fut2mode[1]]
                rel_head_fut = wrap_angle(
                    head_fut_flat[edge_fut2mode[0]] - head_mode_flat[edge_fut2mode[1]]
                )
                rel_dt_fut = anchor_fut_flat[edge_fut2mode[0]] - anchor_mode[edge_fut2mode[1]]
                r_fut2mode = torch.stack([
                    torch.norm(rel_pos_fut[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_mode_flat[edge_fut2mode[1]],
                        nbr_vector=rel_pos_fut[:, :2],
                    ),
                    rel_head_fut,
                    rel_dt_fut.float(),
                ], dim=-1)
                r_fut2mode = self.r_t_emb(continuous_inputs=r_fut2mode, categorical_embs=None)
            else:
                r_fut2mode = torch.empty(0, self.hidden_dim, device=device, dtype=dtype)

            hist_offset = num_nodes * num_hist_chunks
            edge_fut2mode = edge_fut2mode.clone()
            edge_fut2mode[0].add_(hist_offset)

            edge_parts.append(edge_fut2mode)
            r_parts.append(r_fut2mode)

        edge_index_t = torch.cat(edge_parts, dim=1) if len(edge_parts) > 1 else edge_parts[0]
        r_t = torch.cat(r_parts, dim=0) if len(r_parts) > 1 else r_parts[0]

        return r_t, edge_index_t

    def encode_trajectory_tokens(
        self,
        token_bundle: Mapping[str, torch.Tensor],
        num_nodes: int,
        num_chunks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step_feat = token_bundle["step_feat"]
        trans_mask = token_bundle["trans_mask"]
        chunk_valid_flat = token_bundle["chunk_valid_flat"]
        agent_type = token_bundle["agent_type_chunk"]

        P = self.chunk_points
        B = step_feat.size(0)
        mu_flat = step_feat.new_zeros(B, self.z_dim)
        logvar_flat = step_feat.new_zeros(B, self.z_dim)
        vae_recon_loss = torch.tensor(0.0, device=step_feat.device)
        vae_kl_loss = torch.tensor(0.0, device=step_feat.device)

        valid_idx = chunk_valid_flat.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            return (
                mu_flat.view(num_nodes, num_chunks, self.z_dim),
                logvar_flat.view(num_nodes, num_chunks, self.z_dim),
                vae_recon_loss,
                vae_kl_loss,
            )

        mu_valid, logvar_valid = self.chunk_encoder(
            step_feat[valid_idx],
            trans_mask[valid_idx],
            agent_type[valid_idx],
        )
        mu_flat[valid_idx] = mu_valid
        logvar_flat[valid_idx] = logvar_valid

        if self.training:
            std = (0.5 * logvar_valid).exp()
            z = mu_valid + torch.randn_like(std) * std
        else:
            z = mu_valid

        vae_out = self.chunk_decoder(z)
        pred_rel_pos = vae_out[4]
        pred_heading = vae_out[2]
        pred_bridge = vae_out[3]

        target = token_bundle["target"].reshape(B, P, 5)[valid_idx]
        target_bridge = token_bundle["target_bridge"].reshape(B, 3)[valid_idx]
        point_mask = token_bundle["step_mask"].reshape(B, P)[valid_idx].float()
        pair_mask = token_bundle["pair_mask"].reshape(B)[valid_idx].float()

        target_pos = target[..., 0:2]
        target_head = target[..., 2]
        point_count = point_mask.sum(dim=1).clamp_min(1.0)
        pair_den = pair_mask.sum().clamp_min(1.0)

        pos_err = (pred_rel_pos - target_pos).pow(2).sum(dim=-1)
        head_err = wrap_angle(pred_heading - target_head).pow(2)
        vae_recon_loss = (
            ((pos_err * point_mask).sum(dim=1) / point_count).mean()
            + ((head_err * point_mask).sum(dim=1) / point_count).mean()
        )

        bridge_pos_loss = (
            ((pred_bridge[:, :2] - target_bridge[:, :2]).pow(2).sum(dim=-1) * pair_mask).sum() / pair_den
        )
        bridge_head_loss = (
            (wrap_angle(pred_bridge[:, 2] - target_bridge[:, 2]).pow(2) * pair_mask).sum() / pair_den
        )
        vae_recon_loss = vae_recon_loss + bridge_pos_loss + bridge_head_loss
        vae_kl_loss = 0.5 * (
            mu_valid.pow(2) + logvar_valid.exp() - 1.0 - logvar_valid
        ).sum(dim=-1).mean()

        return (
            mu_flat.view(num_nodes, num_chunks, self.z_dim),
            logvar_flat.view(num_nodes, num_chunks, self.z_dim),
            vae_recon_loss,
            vae_kl_loss,
        )

    def forward(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        active_future_chunks: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        device = data["agent"]["position"].device
        num_nodes = int(data["agent"]["num_nodes"])
        num_chunks = self.num_chunks
        chunk_anchor = self.chunk_anchor.to(device)
        num_rollout_chunks = self.num_future_chunks if active_future_chunks is None else max(
            1, min(int(active_future_chunks), self.num_future_chunks)
        )

        if num_chunks < 2:
            raise ValueError(
                f"ChunkAgent requires at least 2 chunks, got {num_chunks}. "
                f"Current chunk split needs history chunks + 1 final chunk."
            )

        mask = data["agent"]["valid_mask"][:, :self.num_historical_steps].contiguous().bool()
        mask = self.repair_valid_mask(mask)

        pos_a = data["agent"]["position"][:, :self.num_historical_steps, :self.input_dim].contiguous()
        head_a = data["agent"]["heading"][:, :self.num_historical_steps].contiguous()
        vel_a = data["agent"]["velocity"][:, :self.num_historical_steps, :self.input_dim].contiguous()
        agent_type = data["agent"]["type"].long()

        pos_pl = data["map_polygon"]["position"][:, :self.input_dim].contiguous()
        orient_pl = data["map_polygon"]["orientation"].contiguous()
        x_pl = map_enc["x_pl"]

        # ==============================================================
        # 0) Unified GT chunk bundle
        # ==============================================================
        total_steps = self.num_historical_steps + self.num_future_steps
        pos_full = data["agent"]["position"][:, :total_steps, :self.input_dim].contiguous()
        head_full = data["agent"]["heading"][:, :total_steps].contiguous()
        vel_full = data["agent"]["velocity"][:, :total_steps, :self.input_dim].contiguous()

        hist_valid_full = self.repair_valid_mask(
            data["agent"]["valid_mask"][:, :self.num_historical_steps].bool()
        )
        future_valid_full = data["agent"]["predict_mask"][
            :, self.num_historical_steps:self.num_historical_steps + self.num_future_steps
        ].bool()
        full_valid = torch.cat([hist_valid_full, future_valid_full], dim=-1)

        future_anchors = chunk_anchor[-1] + self.chunk_stride * torch.arange(
            1, self.num_future_chunks + 1, device=device
        )
        all_anchors = torch.cat([chunk_anchor, future_anchors], dim=0)

        num_all_chunks = int(all_anchors.numel())
        num_hist_anchor_chunks = int(chunk_anchor.numel())   # includes current chunk

        gt_bundle = self.build_chunk_bundle_from_anchors(
            pos_a=pos_full[:, :, :2],
            head_a=head_full,
            vel_a=vel_full[:, :, :2],
            mask=full_valid,
            agent_type=agent_type,
            anchor_idx=all_anchors,
        )

        gt_chunk_valid = gt_bundle["chunk_valid"]            # [N, C_all]

        gt_target_all = gt_bundle["target"]                  # [N, C_all, P, 5]
        gt_target_bridge_all = gt_bundle["target_bridge"]    # [N, C_all, 3]
        gt_step_mask_all = gt_bundle["step_mask"]            # [N, C_all, P]
        gt_pair_mask_all = gt_bundle["pair_mask"]            # [N, C_all]

        mu_gt_all, _, vae_recon_loss, vae_kl_loss = self.encode_trajectory_tokens(
            token_bundle=gt_bundle,
            num_nodes=num_nodes,
            num_chunks=num_all_chunks,
        )
        mu_gt_all_detached = mu_gt_all.detach()
        gt_chunk_valid_all = gt_chunk_valid

        mu_gt_hist_anchors = mu_gt_all_detached[:, :num_hist_anchor_chunks]      # [N, C_hist_all, z_dim]
        mu_gt_fut_anchors = mu_gt_all_detached[:, num_hist_anchor_chunks:]       # [N, K_fut, z_dim]
        valid_gt_hist = gt_chunk_valid_all[:, :num_hist_anchor_chunks]           # [N, C_hist_all]
        valid_gt_fut = gt_chunk_valid_all[:, num_hist_anchor_chunks:]            # [N, K_fut]

        # ==============================================================
        # 1) Reuse historical chunk encodings
        # ==============================================================
        mu_hist_reuse = mu_gt_all[:, :num_hist_anchor_chunks].reshape(-1, self.z_dim)  # [N*C_hist_all, z_dim]

        chunk_valid = gt_chunk_valid_all[:, :num_hist_anchor_chunks]   # [N, C_hist_all]
        chunk_valid_flat = chunk_valid.reshape(-1)

        x_c_flat = mu_hist_reuse.new_zeros(
            num_nodes * num_hist_anchor_chunks, self.hidden_dim
        )
        valid_idx = chunk_valid_flat.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() > 0:
            x_c_flat[valid_idx] = self.chunk_proj(mu_hist_reuse[valid_idx])

        x_chunk0 = x_c_flat.view(num_nodes, num_hist_anchor_chunks, self.hidden_dim)

        num_hist_chunks = num_chunks - 1
        chunk_anchor_hist = chunk_anchor[:-1]
        chunk_anchor_curr = chunk_anchor[-1]

        x_hist = x_chunk0[:, :-1].contiguous()
        x_curr_base = x_chunk0[:, -1].contiguous()

        chunk_valid_hist = chunk_valid[:, :-1].contiguous()
        chunk_valid_curr = chunk_valid[:, -1].contiguous()

        pos_hist = pos_a[:, chunk_anchor_hist, :self.input_dim].contiguous()
        head_hist = head_a[:, chunk_anchor_hist].contiguous()
        head_vector_hist = torch.stack([head_hist.cos(), head_hist.sin()], dim=-1)

        pos_curr = pos_a[:, chunk_anchor_curr, :self.input_dim].contiguous()
        head_curr = head_a[:, chunk_anchor_curr].contiguous()

        num_pl = pos_pl.size(0)

        if isinstance(data, Batch):
            batch_agent_base = data["agent"]["batch"]
            batch_pl_base = data["map_polygon"]["batch"]
            num_graphs = int(data.num_graphs)
        else:
            batch_agent_base = torch.zeros(num_nodes, dtype=torch.long, device=device)
            batch_pl_base = torch.zeros(num_pl, dtype=torch.long, device=device)
            num_graphs = 1

        # ==============================================================
        # 2) History branch
        # ==============================================================
        hist_aux_rel_pos_list: List[torch.Tensor] = []
        hist_aux_heading_list: List[torch.Tensor] = []
        hist_aux_vel_list: List[torch.Tensor] = []
        hist_aux_bridge_list: List[torch.Tensor] = []
        hist_aux_z_list: List[torch.Tensor] = []

        pos_t = pos_hist.reshape(-1, self.input_dim)
        head_t = head_hist.reshape(-1)
        head_vector_t = head_vector_hist.reshape(-1, 2)

        mask_t = chunk_valid_hist.unsqueeze(2) & chunk_valid_hist.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]

        if edge_index_t.size(1) > 0:
            rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
            rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
            chunk_id_per_flat_t = torch.arange(num_hist_chunks, device=device).repeat(num_nodes)
            anchor_per_flat_t = chunk_anchor_hist[chunk_id_per_flat_t]
            rel_dt = anchor_per_flat_t[edge_index_t[0]] - anchor_per_flat_t[edge_index_t[1]]
            r_t = torch.stack([
                torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]],
                    nbr_vector=rel_pos_t[:, :2],
                ),
                rel_head_t,
                rel_dt.float(),
            ], dim=-1)
            r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        else:
            r_t = torch.empty(0, self.hidden_dim, device=device, dtype=x_hist.dtype)

        pos_s_hist = pos_hist.transpose(0, 1).reshape(-1, self.input_dim)
        head_s_hist = head_hist.transpose(0, 1).reshape(-1)
        head_vector_s_hist = head_vector_hist.transpose(0, 1).reshape(-1, 2)
        mask_s_hist = chunk_valid_hist.transpose(0, 1).reshape(-1)

        batch_s_pl2c_hist = batch_agent_base.repeat(num_hist_chunks)
        edge_index_pl2c_hist = radius(
            x=pos_s_hist[:, :2], y=pos_pl[:, :2],
            r=self.pl2a_radius, batch_x=batch_s_pl2c_hist, batch_y=batch_pl_base,
            max_num_neighbors=300,
        )
        edge_index_pl2c_hist = edge_index_pl2c_hist[:, mask_s_hist[edge_index_pl2c_hist[1]]]

        if edge_index_pl2c_hist.size(1) > 0:
            rel_pos_pl2c_hist = pos_pl[edge_index_pl2c_hist[0]] - pos_s_hist[edge_index_pl2c_hist[1]]
            rel_orient_pl2c_hist = wrap_angle(
                orient_pl[edge_index_pl2c_hist[0]] - head_s_hist[edge_index_pl2c_hist[1]]
            )
            r_pl2c_hist = torch.stack([
                torch.norm(rel_pos_pl2c_hist[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s_hist[edge_index_pl2c_hist[1]],
                    nbr_vector=rel_pos_pl2c_hist[:, :2],
                ),
                rel_orient_pl2c_hist,
            ], dim=-1)
            r_pl2c_hist = self.r_pl2c_emb(continuous_inputs=r_pl2c_hist, categorical_embs=None)
        else:
            r_pl2c_hist = torch.empty(0, self.hidden_dim, device=device, dtype=x_hist.dtype)

        chunk_id_hist = torch.arange(num_hist_chunks, device=device).repeat_interleave(num_nodes)
        batch_s_c2c_hist = batch_agent_base.repeat(num_hist_chunks) + chunk_id_hist * num_graphs

        edge_index_c2c_hist = radius_graph(
            x=pos_s_hist[:, :2], r=self.a2a_radius,
            batch=batch_s_c2c_hist, loop=False, max_num_neighbors=300,
        )
        edge_index_c2c_hist = subgraph(subset=mask_s_hist, edge_index=edge_index_c2c_hist)[0]

        if edge_index_c2c_hist.size(1) > 0:
            rel_pos_c2c_hist = pos_s_hist[edge_index_c2c_hist[0]] - pos_s_hist[edge_index_c2c_hist[1]]
            rel_head_c2c_hist = wrap_angle(
                head_s_hist[edge_index_c2c_hist[0]] - head_s_hist[edge_index_c2c_hist[1]]
            )
            r_c2c_hist = torch.stack([
                torch.norm(rel_pos_c2c_hist[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s_hist[edge_index_c2c_hist[1]],
                    nbr_vector=rel_pos_c2c_hist[:, :2],
                ),
                rel_head_c2c_hist,
            ], dim=-1)
            r_c2c_hist = self.r_c2c_emb(continuous_inputs=r_c2c_hist, categorical_embs=None)
        else:
            r_c2c_hist = torch.empty(0, self.hidden_dim, device=device, dtype=x_hist.dtype)

        edge_index_hist_m2m = self.build_self_loop_edges(mask_s_hist)

        x_hist = x_hist + self.hist_emb
        hist_k_cache: List[torch.Tensor] = []
        hist_v_cache: List[torch.Tensor] = []
        map_k_cache: List[torch.Tensor] = []
        map_v_cache: List[torch.Tensor] = []
        for i in range(self.num_layers):
            x_hist_flat_am = x_hist.reshape(-1, self.hidden_dim)
            hist_k_i, hist_v_i = self.t_attn_layers[i].project_kv(x_hist_flat_am)
            hist_k_cache.append(hist_k_i)
            hist_v_cache.append(hist_v_i)
            x_hist_flat_am = self.t_attn_layers[i](x_hist_flat_am, r_t, edge_index_t)
            x_hist_flat_cm = x_hist_flat_am.view(num_nodes, num_hist_chunks, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            map_k_i, map_v_i = self.pl2c_attn_layers[i].project_kv(x_pl)
            map_k_cache.append(map_k_i)
            map_v_cache.append(map_v_i)
            x_hist_flat_cm = self.pl2c_attn_layers[i].forward_with_kv(
                x_hist_flat_cm, map_k_i, map_v_i, r_pl2c_hist, edge_index_pl2c_hist
            )
            x_hist_flat_cm = self.c2c_attn_layers[i](x_hist_flat_cm, r_c2c_hist, edge_index_c2c_hist)
            if i == self.num_layers - 1:
                x_hist_flat_cm = self.m2m_attn_layer(x_hist_flat_cm, None, edge_index_hist_m2m)
            x_hist = x_hist_flat_cm.view(num_hist_chunks, num_nodes, self.hidden_dim).transpose(0, 1).contiguous()

            if i in self.aux_layer_ids:
                z_hist_aux = self.chunk_proj_back(x_hist.reshape(-1, self.hidden_dim))
                hist_aux_out = self.chunk_decoder(z_hist_aux)

                hist_aux_rel_pos_list.append(
                    hist_aux_out[4].view(num_nodes, num_hist_chunks, self.chunk_points, 2)
                )
                hist_aux_heading_list.append(
                    hist_aux_out[2].view(num_nodes, num_hist_chunks, self.chunk_points)
                )
                hist_aux_vel_list.append(
                    hist_aux_out[1].view(num_nodes, num_hist_chunks, self.chunk_points, 2)
                )
                hist_aux_bridge_list.append(
                    hist_aux_out[3].view(num_nodes, num_hist_chunks, 3)
                )
                hist_aux_z_list.append(
                    z_hist_aux.view(num_nodes, num_hist_chunks, self.z_dim)
                )

        x_hist = x_hist * chunk_valid_hist.unsqueeze(-1)
        z_hist_pred = self.chunk_proj_back(x_hist.reshape(-1, self.hidden_dim))
        hist_out = self.chunk_decoder(z_hist_pred)

        hist_delta = hist_out[0].view(num_nodes, num_hist_chunks, self.chunk_points, 2)[..., :-1, :]
        hist_vel = hist_out[1].view(num_nodes, num_hist_chunks, self.chunk_points, 2)
        hist_heading_local = hist_out[2].view(num_nodes, num_hist_chunks, self.chunk_points)
        hist_bridge = hist_out[3].view(num_nodes, num_hist_chunks, 3)
        hist_rel_pos = hist_out[4].view(num_nodes, num_hist_chunks, self.chunk_points, 2)

        next_pos_hist = pos_a[:, chunk_anchor[1:], :self.input_dim]
        next_head_hist = head_a[:, chunk_anchor[1:]]

        hist_pos_global = next_pos_hist.unsqueeze(-2) + self.local_vec_to_global(hist_rel_pos, next_head_hist)
        hist_heading_global = wrap_angle(next_head_hist.unsqueeze(-1) + hist_heading_local)

        hist_chunk_mask_4d = chunk_valid_hist.unsqueeze(-1).unsqueeze(-1)
        hist_chunk_mask_3d = chunk_valid_hist.unsqueeze(-1)

        hist_rel_pos = torch.where(
            hist_chunk_mask_4d.expand_as(hist_rel_pos), hist_rel_pos, torch.zeros_like(hist_rel_pos)
        )
        hist_delta = torch.where(
            hist_chunk_mask_4d.expand_as(hist_delta), hist_delta, torch.zeros_like(hist_delta)
        )
        hist_pos_global = torch.where(
            hist_chunk_mask_4d.expand_as(hist_pos_global),
            hist_pos_global,
            next_pos_hist.unsqueeze(-2).expand_as(hist_pos_global),
        )
        hist_heading_local = torch.where(
            hist_chunk_mask_3d.expand_as(hist_heading_local), hist_heading_local, torch.zeros_like(hist_heading_local)
        )
        hist_heading_global = torch.where(
            hist_chunk_mask_3d.expand_as(hist_heading_global),
            hist_heading_global,
            next_head_hist.unsqueeze(-1).expand_as(hist_heading_global),
        )

        # ==============================================================
        # 3) Mode branch init
        # ==============================================================
        x_mode = x_curr_base[:, None, :] + self.mode_emb.weight[None, :, :]
        mode_valid = chunk_valid_curr[:, None].expand(-1, self.num_modes)

        pos_mode = pos_curr[:, None, :].expand(-1, self.num_modes, -1).contiguous()
        head_mode = head_curr[:, None].expand(-1, self.num_modes).contiguous()

        current_anchor_idx = int(chunk_anchor_curr.item())

        # ==============================================================
        # 4) Future memory banks
        # ==============================================================
        future_aux_rel_pos_dict: Dict[int, List[torch.Tensor]] = {i: [] for i in self.aux_layer_ids}
        future_aux_heading_dict: Dict[int, List[torch.Tensor]] = {i: [] for i in self.aux_layer_ids}
        future_aux_vel_dict: Dict[int, List[torch.Tensor]] = {i: [] for i in self.aux_layer_ids}
        future_aux_bridge_dict: Dict[int, List[torch.Tensor]] = {i: [] for i in self.aux_layer_ids}
        future_aux_pos_global_dict: Dict[int, List[torch.Tensor]] = {i: [] for i in self.aux_layer_ids}
        future_aux_heading_global_dict: Dict[int, List[torch.Tensor]] = {i: [] for i in self.aux_layer_ids}
        future_aux_z_dict: Dict[int, List[torch.Tensor]] = {i: [] for i in self.aux_layer_ids}

        fut_k_cache: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        fut_v_cache: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        fut_pos_list: List[torch.Tensor] = []
        fut_head_list: List[torch.Tensor] = []
        fut_valid_list: List[torch.Tensor] = []
        fut_anchor_list: List[int] = []

        future_anchor_pos_list: List[torch.Tensor] = []
        future_anchor_heading_list: List[torch.Tensor] = []
        future_rel_pos_list: List[torch.Tensor] = []
        future_heading_local_list: List[torch.Tensor] = []
        future_delta_list: List[torch.Tensor] = []
        future_pos_global_list: List[torch.Tensor] = []
        future_heading_global_list: List[torch.Tensor] = []
        future_anchor_idx_list: List[int] = []
        future_vel_list: List[torch.Tensor] = []
        future_bridge_list: List[torch.Tensor] = []
        future_z_list: List[torch.Tensor] = []

        # ==============================================================
        # 5) Future rollout
        # ==============================================================
        for rollout_idx in range(num_rollout_chunks):
            curr_pos_mode = pos_mode
            curr_head_mode = head_mode
            curr_anchor_idx = current_anchor_idx

            (
                pos_mode_flat, head_mode_flat, head_vector_mode_flat, mode_valid_flat,
                edge_index_pl2m, r_pl2m,
                edge_index_mode_c2c, r_mode_c2c,
                edge_index_mode_m2m,
            ) = self.build_mode_spatial_inputs(
                pos_mode=curr_pos_mode, head_mode=curr_head_mode,
                mode_valid=mode_valid, batch_agent_base=batch_agent_base,
                batch_pl_base=batch_pl_base, num_graphs=num_graphs,
                pos_pl=pos_pl, orient_pl=orient_pl, dtype=x_mode.dtype,
            )

            r_t_mode, edge_index_t_mode = self.build_temporal_geometry_to_mode_inputs(
                pos_hist=pos_hist,
                head_hist=head_hist,
                valid_hist=chunk_valid_hist,
                anchor_hist=chunk_anchor_hist,
                fut_pos_list=fut_pos_list,
                fut_head_list=fut_head_list,
                fut_valid_list=fut_valid_list,
                fut_anchor_list=fut_anchor_list,
                pos_mode=curr_pos_mode,
                head_mode=curr_head_mode,
                mode_valid=mode_valid,
                current_anchor=curr_anchor_idx,
                dtype=x_mode.dtype,
            )

            step_k_cache: List[torch.Tensor] = []
            step_v_cache: List[torch.Tensor] = []

            for i in range(self.num_layers):
                x_mode_flat = x_mode.reshape(num_nodes * self.num_modes, self.hidden_dim)
                if rollout_idx + 1 < num_rollout_chunks:
                    step_k_i, step_v_i = self.t_attn_layers[i].project_kv(x_mode_flat)
                    step_k_cache.append(
                        step_k_i.view(num_nodes, self.num_modes, self.num_heads, self.head_dim)
                    )
                    step_v_cache.append(
                        step_v_i.view(num_nodes, self.num_modes, self.num_heads, self.head_dim)
                    )

                k_mem_parts = [hist_k_cache[i]]
                v_mem_parts = [hist_v_cache[i]]
                if fut_k_cache[i]:
                    num_cached_steps = len(fut_k_cache[i])
                    k_mem_parts.append(
                        torch.stack(fut_k_cache[i], dim=2).reshape(
                            num_nodes * self.num_modes * num_cached_steps,
                            self.num_heads,
                            self.head_dim,
                        )
                    )
                    v_mem_parts.append(
                        torch.stack(fut_v_cache[i], dim=2).reshape(
                            num_nodes * self.num_modes * num_cached_steps,
                            self.num_heads,
                            self.head_dim,
                        )
                    )

                x_mode_flat = self.t_attn_layers[i].forward_with_kv(
                    x_mode_flat,
                    torch.cat(k_mem_parts, dim=0),
                    torch.cat(v_mem_parts, dim=0),
                    r_t_mode,
                    edge_index_t_mode,
                )
                x_mode_flat = self.pl2c_attn_layers[i].forward_with_kv(
                    x_mode_flat, map_k_cache[i], map_v_cache[i], r_pl2m, edge_index_pl2m
                )
                x_mode_flat = self.c2c_attn_layers[i](x_mode_flat, r_mode_c2c, edge_index_mode_c2c)
                if i == self.num_layers - 1:
                    x_mode_flat = self.m2m_attn_layer(x_mode_flat, None, edge_index_mode_m2m)
                x_mode = x_mode_flat.view(num_nodes, self.num_modes, self.hidden_dim)

                if i in self.aux_layer_ids:
                    z_mode_aux = self.chunk_proj_back(x_mode.reshape(-1, self.hidden_dim))
                    fut_aux_out = self.chunk_decoder(z_mode_aux)

                    aux_delta = fut_aux_out[0].view(num_nodes, self.num_modes, self.chunk_points, 2)[..., :-1, :]
                    aux_vel = fut_aux_out[1].view(num_nodes, self.num_modes, self.chunk_points, 2)
                    aux_heading_local = fut_aux_out[2].view(num_nodes, self.num_modes, self.chunk_points)
                    aux_bridge = fut_aux_out[3].view(num_nodes, self.num_modes, 3)
                    aux_rel_pos = fut_aux_out[4].view(num_nodes, self.num_modes, self.chunk_points, 2)

                    # auxiliary global chunk geometry
                    aux_disp_to_next_anchor_local = aux_delta.sum(dim=-2)
                    aux_new_anchor_heading = wrap_angle(curr_head_mode - aux_heading_local[..., 0])
                    aux_disp_to_next_anchor_global = self.local_vec_to_global(
                        aux_disp_to_next_anchor_local, aux_new_anchor_heading
                    )
                    aux_new_anchor_pos = curr_pos_mode + aux_disp_to_next_anchor_global

                    aux_new_anchor_pos = torch.where(mode_valid.unsqueeze(-1), aux_new_anchor_pos, curr_pos_mode)
                    aux_new_anchor_heading = torch.where(mode_valid, aux_new_anchor_heading, curr_head_mode)

                    aux_pos_global = aux_new_anchor_pos.unsqueeze(-2) + self.local_vec_to_global(
                        aux_rel_pos, aux_new_anchor_heading
                    )
                    aux_heading_global = wrap_angle(aux_new_anchor_heading.unsqueeze(-1) + aux_heading_local)

                    future_aux_rel_pos_dict[i].append(aux_rel_pos)
                    future_aux_heading_dict[i].append(aux_heading_local)
                    future_aux_vel_dict[i].append(aux_vel)
                    future_aux_bridge_dict[i].append(aux_bridge)
                    future_aux_pos_global_dict[i].append(aux_pos_global)
                    future_aux_heading_global_dict[i].append(aux_heading_global)
                    future_aux_z_dict[i].append(
                        z_mode_aux.view(num_nodes, self.num_modes, self.z_dim)
                    )

            z_mode = self.chunk_proj_back(x_mode.view(-1, self.hidden_dim))
            z_mode_for_decode = z_mode
            x_mode = self.chunk_proj(z_mode).view(num_nodes, self.num_modes, self.hidden_dim) \
                + self.mode_emb.weight[None, :, :]
            x_mode = x_mode * mode_valid.unsqueeze(-1)

            future_z_list.append(z_mode.view(num_nodes, self.num_modes, self.z_dim))

            out = self.chunk_decoder(z_mode_for_decode)
            pred_delta = out[0].view(num_nodes, self.num_modes, self.chunk_points, 2)[..., :-1, :]
            pred_vel = out[1].view(num_nodes, self.num_modes, self.chunk_points, 2)
            pred_heading_local = out[2].view(num_nodes, self.num_modes, self.chunk_points)
            pred_bridge = out[3].view(num_nodes, self.num_modes, 3)
            pred_rel_pos = out[4].view(num_nodes, self.num_modes, self.chunk_points, 2)

            disp_to_next_anchor_local = pred_delta.sum(dim=-2)
            new_anchor_heading = wrap_angle(curr_head_mode - pred_heading_local[..., 0])
            disp_to_next_anchor_global = self.local_vec_to_global(disp_to_next_anchor_local, new_anchor_heading)
            new_anchor_pos = curr_pos_mode + disp_to_next_anchor_global

            new_anchor_pos = torch.where(mode_valid.unsqueeze(-1), new_anchor_pos, curr_pos_mode)
            new_anchor_heading = torch.where(mode_valid, new_anchor_heading, curr_head_mode)

            pred_pos_global = new_anchor_pos.unsqueeze(-2) + self.local_vec_to_global(pred_rel_pos, new_anchor_heading)
            pred_heading_global = wrap_angle(new_anchor_heading.unsqueeze(-1) + pred_heading_local)

            next_anchor_idx = curr_anchor_idx + self.chunk_stride

            future_anchor_pos_list.append(new_anchor_pos)
            future_anchor_heading_list.append(new_anchor_heading)
            future_rel_pos_list.append(pred_rel_pos)
            future_heading_local_list.append(pred_heading_local)
            future_delta_list.append(pred_delta)
            future_pos_global_list.append(pred_pos_global)
            future_heading_global_list.append(pred_heading_global)
            future_anchor_idx_list.append(next_anchor_idx)
            future_vel_list.append(pred_vel)
            future_bridge_list.append(pred_bridge)

            if rollout_idx + 1 < num_rollout_chunks:
                for i in range(self.num_layers):
                    fut_k_cache[i].append(step_k_cache[i])
                    fut_v_cache[i].append(step_v_cache[i])
                fut_pos_list.append(curr_pos_mode)
                fut_head_list.append(curr_head_mode)
                fut_valid_list.append(mode_valid)
                fut_anchor_list.append(curr_anchor_idx)

            pos_mode = new_anchor_pos.detach()
            head_mode = new_anchor_heading.detach()
            current_anchor_idx = next_anchor_idx

        future_anchor_pos = torch.stack(future_anchor_pos_list, dim=2)
        future_anchor_heading = torch.stack(future_anchor_heading_list, dim=2)
        future_rel_pos = torch.stack(future_rel_pos_list, dim=2)
        future_heading_local = torch.stack(future_heading_local_list, dim=2)
        future_delta = torch.stack(future_delta_list, dim=2)
        future_pos_global = torch.stack(future_pos_global_list, dim=2)
        future_heading_global = torch.stack(future_heading_global_list, dim=2)
        future_anchor_index = torch.tensor(future_anchor_idx_list, dtype=chunk_anchor.dtype, device=device)
        future_vel = torch.stack(future_vel_list, dim=2)
        future_bridge = torch.stack(future_bridge_list, dim=2)
        future_z = torch.stack(future_z_list, dim=2)

        if len(self.aux_layer_ids) > 0:
            aux_hist_rel_pos = torch.stack(hist_aux_rel_pos_list, dim=0)      # [L_aux, N, C_hist, P, 2]
            aux_hist_heading = torch.stack(hist_aux_heading_list, dim=0)      # [L_aux, N, C_hist, P]
            aux_hist_vel = torch.stack(hist_aux_vel_list, dim=0)              # [L_aux, N, C_hist, P, 2]
            aux_hist_bridge = torch.stack(hist_aux_bridge_list, dim=0)        # [L_aux, N, C_hist, 3]
            aux_hist_z = torch.stack(hist_aux_z_list, dim=0)                  # [L_aux, N, C_hist, z_dim]

            aux_hist_pos_global = next_pos_hist.unsqueeze(0).unsqueeze(-2) + self.local_vec_to_global(
                aux_hist_rel_pos, next_head_hist.unsqueeze(0)
            )
            aux_hist_heading_global = wrap_angle(
                next_head_hist.unsqueeze(0).unsqueeze(-1) + aux_hist_heading
            )
            aux_hist_traj = aux_hist_pos_global[:, :, :, 1:, :].contiguous().reshape(
                len(self.aux_layer_ids), num_nodes, num_hist_chunks * self.chunk_stride, 2
            )
            aux_hist_heading_traj = aux_hist_heading_global[:, :, :, 1:].contiguous().reshape(
                len(self.aux_layer_ids), num_nodes, num_hist_chunks * self.chunk_stride
            )

            aux_future_pos_global = torch.stack([
                torch.stack(future_aux_pos_global_dict[i], dim=2) for i in self.aux_layer_ids
            ], dim=0)  # [L_aux, N, M, K, P, 2]

            aux_future_heading_global = torch.stack([
                torch.stack(future_aux_heading_global_dict[i], dim=2) for i in self.aux_layer_ids
            ], dim=0)  # [L_aux, N, M, K, P]

            aux_future_rel_pos = torch.stack([
                torch.stack(future_aux_rel_pos_dict[i], dim=2) for i in self.aux_layer_ids
            ], dim=0)                                                         # [L_aux, N, M, K, P, 2]

            aux_future_heading = torch.stack([
                torch.stack(future_aux_heading_dict[i], dim=2) for i in self.aux_layer_ids
            ], dim=0)                                                         # [L_aux, N, M, K, P]

            aux_future_vel = torch.stack([
                torch.stack(future_aux_vel_dict[i], dim=2) for i in self.aux_layer_ids
            ], dim=0)                                                         # [L_aux, N, M, K, P, 2]

            aux_future_bridge = torch.stack([
                torch.stack(future_aux_bridge_dict[i], dim=2) for i in self.aux_layer_ids
            ], dim=0)                                                         # [L_aux, N, M, K, 3]

            aux_future_z = torch.stack([
                torch.stack(future_aux_z_dict[i], dim=2) for i in self.aux_layer_ids
            ], dim=0)                                                         # [L_aux, N, M, K, z_dim]

            aux_future_traj_full = aux_future_pos_global[:, :, :, :, 1:, :].contiguous().reshape(
                len(self.aux_layer_ids), num_nodes, self.num_modes,
                num_rollout_chunks * self.chunk_stride, 2
            )
            aux_future_heading_traj_full = aux_future_heading_global[:, :, :, :, 1:].contiguous().reshape(
                len(self.aux_layer_ids), num_nodes, self.num_modes,
                num_rollout_chunks * self.chunk_stride
            )

            aux_future_traj = aux_future_traj_full[:, :, :, :self.num_future_steps, :]
            aux_future_heading_traj = aux_future_heading_traj_full[:, :, :, :self.num_future_steps]
        else:
            aux_hist_rel_pos = x_hist.new_empty(0)
            aux_hist_heading = x_hist.new_empty(0)
            aux_hist_vel = x_hist.new_empty(0)
            aux_hist_bridge = x_hist.new_empty(0)
            aux_hist_z = x_hist.new_empty(0)
            aux_hist_pos_global = x_hist.new_empty(0)
            aux_hist_heading_global = x_hist.new_empty(0)
            aux_hist_traj = x_hist.new_empty(0)
            aux_hist_heading_traj = x_hist.new_empty(0)

            aux_future_rel_pos = x_mode.new_empty(0)
            aux_future_heading = x_mode.new_empty(0)
            aux_future_vel = x_mode.new_empty(0)
            aux_future_bridge = x_mode.new_empty(0)
            aux_future_pos_global = x_mode.new_empty(0)
            aux_future_heading_global = x_mode.new_empty(0)
            aux_future_traj = x_mode.new_empty(0)
            aux_future_heading_traj = x_mode.new_empty(0)
            aux_future_z = x_mode.new_empty(0)

        history_traj = hist_pos_global[:, :, 1:, :].contiguous().reshape(
            num_nodes, num_hist_chunks * self.chunk_stride, 2
        )
        history_heading_traj = hist_heading_global[:, :, 1:].contiguous().reshape(
            num_nodes, num_hist_chunks * self.chunk_stride
        )
        future_traj_full = future_pos_global[:, :, :, 1:, :].contiguous().reshape(
            num_nodes, self.num_modes, num_rollout_chunks * self.chunk_stride, 2
        )
        future_heading_traj_full = future_heading_global[:, :, :, 1:].contiguous().reshape(
            num_nodes, self.num_modes, num_rollout_chunks * self.chunk_stride
        )
        future_traj = future_traj_full[:, :, :self.num_future_steps, :]
        future_heading_traj = future_heading_traj_full[:, :, :self.num_future_steps]

        mode_logits = self.prob_head(x_mode).squeeze(-1)
        mode_logits = mode_logits.masked_fill(~mode_valid, -1e9)
        mode_prob = torch.softmax(mode_logits, dim=-1)

        return {
            "x_hist": x_hist,
            "x_mode": x_mode,

            "hist_delta": hist_delta,
            "hist_rel_pos": hist_rel_pos,
            "hist_pos_global": hist_pos_global,
            "hist_heading_local": hist_heading_local,
            "hist_heading_global": hist_heading_global,
            "hist_vel": hist_vel,
            "hist_bridge": hist_bridge,
            "hist_source_valid": chunk_valid_hist,

            "future_anchor_pos": future_anchor_pos,
            "future_anchor_heading": future_anchor_heading,
            "future_anchor_index": future_anchor_index,
            "future_rel_pos": future_rel_pos,
            "future_heading_local": future_heading_local,
            "future_delta": future_delta,
            "future_pos_global": future_pos_global,
            "future_heading_global": future_heading_global,
            "future_vel": future_vel,
            "future_bridge": future_bridge,
            "future_z": future_z,

            "history_traj": history_traj,
            "history_heading_traj": history_heading_traj,
            "future_traj": future_traj,
            "future_heading_traj": future_heading_traj,

            "mode_logits": mode_logits,
            "mode_prob": mode_prob,

            "vae_recon_loss": vae_recon_loss,
            "vae_kl_loss": vae_kl_loss,

            "z_hist_pred": z_hist_pred.view(num_nodes, num_hist_chunks, self.z_dim),
            "mu_gt_hist_targets": mu_gt_hist_anchors[:, 1:],   # [N, C_hist, z_dim]
            "mu_gt_fut": mu_gt_fut_anchors,                    # [N, K_fut, z_dim]
            "valid_gt_hist_targets": valid_gt_hist[:, 1:],    # [N, C_hist]
            "valid_gt_fut": valid_gt_fut,                     # [N, K_fut]

            "gt_hist_target": gt_target_all[:, 1:num_hist_anchor_chunks],               # [N, C_hist, P, 5]
            "gt_hist_target_bridge": gt_target_bridge_all[:, 1:num_hist_anchor_chunks], # [N, C_hist, 3]
            "gt_hist_step_mask": gt_step_mask_all[:, 1:num_hist_anchor_chunks],         # [N, C_hist, P]
            "gt_hist_pair_mask": gt_pair_mask_all[:, 1:num_hist_anchor_chunks],         # [N, C_hist]

            "gt_fut_target": gt_target_all[:, num_hist_anchor_chunks:],                 # [N, K_fut, P, 5]
            "gt_fut_target_bridge": gt_target_bridge_all[:, num_hist_anchor_chunks:],   # [N, K_fut, 3]
            "gt_fut_step_mask": gt_step_mask_all[:, num_hist_anchor_chunks:],           # [N, K_fut, P]
            "gt_fut_pair_mask": gt_pair_mask_all[:, num_hist_anchor_chunks:],           # [N, K_fut]

            "aux_hist_rel_pos": aux_hist_rel_pos,
            "aux_hist_heading": aux_hist_heading,
            "aux_hist_vel": aux_hist_vel,
            "aux_hist_bridge": aux_hist_bridge,
            "aux_hist_z": aux_hist_z,
            "aux_hist_pos_global": aux_hist_pos_global,
            "aux_hist_heading_global": aux_hist_heading_global,
            "aux_hist_traj": aux_hist_traj,
            "aux_hist_heading_traj": aux_hist_heading_traj,

            "aux_future_rel_pos": aux_future_rel_pos,
            "aux_future_heading": aux_future_heading,
            "aux_future_vel": aux_future_vel,
            "aux_future_bridge": aux_future_bridge,
            "aux_future_pos_global": aux_future_pos_global,
            "aux_future_heading_global": aux_future_heading_global,
            "aux_future_traj": aux_future_traj,
            "aux_future_heading_traj": aux_future_heading_traj,
            "aux_future_z": aux_future_z,
        }

    def compute_all_losses(self, pred, data, config):
        device = pred["x_mode"].device
        num_nodes = pred["x_mode"].size(0)
        aux_last_scale = config["aux_last_layer_scale"]
        aux_decay = config["aux_layer_decay"]
        active_future_chunks = pred["future_rel_pos"].size(2)
        active_future_steps = pred["future_traj"].size(2)

        total_steps = self.num_historical_steps + self.num_future_steps
        pos_full = data["agent"]["position"][:, :total_steps, :2].contiguous()
        head_full = data["agent"]["heading"][:, :total_steps].contiguous()
        future_valid = data["agent"]["predict_mask"][
            :, self.num_historical_steps:self.num_historical_steps + active_future_steps
        ].bool()
        chunk_anchor = self.chunk_anchor.to(device)

        bw = dict(
            bridge_weight_alpha=config["bridge_weight_alpha"],
            bridge_weight_angle_ref=config["bridge_weight_angle_ref"],
            bridge_weight_dist_thr=config["bridge_weight_dist_thr"],
        )

        aux_hist_chunk_loss = torch.tensor(0.0, device=device)
        aux_hist_traj_loss = torch.tensor(0.0, device=device)
        aux_hist_heading_traj_loss = torch.tensor(0.0, device=device)
        aux_fut_chunk_loss = torch.tensor(0.0, device=device)
        aux_fut_traj_wta_loss = torch.tensor(0.0, device=device)
        aux_fut_heading_traj_loss = torch.tensor(0.0, device=device)
        aux_z_align_hist = torch.tensor(0.0, device=device)
        aux_z_align_fut = torch.tensor(0.0, device=device)

        # ----- 1) History chunk recon -----
        hist_target = pred["gt_hist_target"]
        hist_target_bridge = pred["gt_hist_target_bridge"]
        hist_step_mask = pred["gt_hist_step_mask"]
        hist_pair_mask = pred["gt_hist_pair_mask"]

        src_valid = pred["hist_source_valid"]
        hist_step_mask = hist_step_mask & src_valid.unsqueeze(-1)
        hist_pair_mask = hist_pair_mask & src_valid

        hist_loss, hist_stats = self.chunk_recon_loss(
            pred["hist_rel_pos"], pred["hist_heading_local"], pred["hist_vel"], pred["hist_bridge"],
            hist_target, hist_target_bridge, hist_step_mask, hist_pair_mask, **bw,
        )
        hist_stats = {f"hist_chunk_{k}": v for k, v in hist_stats.items()}

        # ----- 1.5) History global trajectory loss -----
        hist_traj_start = int(chunk_anchor[1].item()) - self.chunk_stride + 1
        hist_traj_end = hist_traj_start + pred["history_traj"].size(1)
        gt_hist_traj = pos_full[:, hist_traj_start:hist_traj_end, :2]
        hist_traj_mask = hist_step_mask[:, :, 1:].reshape(num_nodes, -1).float()

        hist_traj_l1 = torch.abs(pred["history_traj"] - gt_hist_traj).sum(dim=-1)
        hist_traj_loss = (hist_traj_l1 * hist_traj_mask).sum(0) / hist_traj_mask.sum(0).clamp_min(1.0)
        hist_traj_loss = hist_traj_loss.mean()

        # ----- 1.6) History global heading trajectory loss -----
        gt_hist_heading_traj = head_full[:, hist_traj_start:hist_traj_end]
        hist_heading_traj_err = wrap_angle(
            pred["history_heading_traj"] - gt_hist_heading_traj
        ).abs()
        hist_heading_traj_loss = (
            hist_heading_traj_err * hist_traj_mask
        ).sum() / hist_traj_mask.sum().clamp_min(1.0)

        # ----- 2) Trajectory-space WTA -> best_mode -----
        gt_future_pos = pos_full[
            :, self.num_historical_steps:self.num_historical_steps + active_future_steps
        ]
        gt_future_heading = head_full[
            :, self.num_historical_steps:self.num_historical_steps + active_future_steps
        ]
        reg_mask = future_valid.float()

        traj_l2 = torch.norm(pred["future_traj"] - gt_future_pos.unsqueeze(1), p=2, dim=-1)
        traj_energy = (traj_l2 * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = traj_energy.argmin(dim=-1)
        batch_idx = torch.arange(num_nodes, device=device)

        # ----- 3) Future chunk recon (winner) -----
        future_target = pred["gt_fut_target"][:, :active_future_chunks]
        future_target_bridge = pred["gt_fut_target_bridge"][:, :active_future_chunks]
        future_step_mask = pred["gt_fut_step_mask"][:, :active_future_chunks]
        future_pair_mask = pred["gt_fut_pair_mask"][:, :active_future_chunks]

        fut_loss, fut_stats = self.chunk_recon_loss(
            pred["future_rel_pos"][batch_idx, best_mode],
            pred["future_heading_local"][batch_idx, best_mode],
            pred["future_vel"][batch_idx, best_mode],
            pred["future_bridge"][batch_idx, best_mode],
            future_target, future_target_bridge, future_step_mask, future_pair_mask, **bw,
        )
        fut_stats = {f"future_chunk_{k}": v for k, v in fut_stats.items()}

        # ----- 4) Trajectory WTA L1 -----
        traj_best = pred["future_traj"][batch_idx, best_mode]
        traj_l1 = torch.abs(traj_best - gt_future_pos).sum(dim=-1)
        traj_wta_loss = (traj_l1 * reg_mask).sum(0) / reg_mask.sum(0).clamp_min(1.0)
        traj_wta_loss = traj_wta_loss.mean()

        # ----- 4.5) Future trajectory heading loss (winner) -----
        heading_best = pred["future_heading_traj"][batch_idx, best_mode]
        future_heading_traj_err = wrap_angle(
            heading_best - gt_future_heading
        ).abs()
        future_heading_traj_loss = (
            future_heading_traj_err * reg_mask
        ).sum() / reg_mask.sum().clamp_min(1.0)

        # ----- 5) Soft classification -----
        cls_mask = reg_mask[:, -1]
        pred_final_d = pred["future_traj"][:, :, -1, :].detach()
        gt_final = gt_future_pos[:, -1]
        cls_l2 = torch.norm(pred_final_d - gt_final.unsqueeze(1), p=2, dim=-1)
        log_pi = F.log_softmax(pred["mode_logits"], dim=-1)
        soft_cls = -torch.logsumexp(log_pi - config["soft_cls_temp"] * cls_l2, dim=-1)
        soft_cls_loss = (soft_cls * cls_mask).sum() / cls_mask.sum().clamp_min(1.0)

        # ----- 6) VAE loss -----
        vae_loss = pred["vae_recon_loss"] + config["vae_kl_weight"] * pred["vae_kl_loss"]

        # ----- 7) Z-alignment -----
        z_hist_pred = pred["z_hist_pred"]           # [N, C_hist, z_dim]
        mu_gt_hist = pred["mu_gt_hist_targets"]     # [N, C_hist, z_dim]
        valid_gh = pred["valid_gt_hist_targets"]    # [N, C_hist]
        hist_align_mask = pred["hist_source_valid"].float() * valid_gh.float()
        hist_align_d = hist_align_mask.sum().clamp_min(1.0)
        z_align_hist = ((z_hist_pred - mu_gt_hist).pow(2).sum(-1) * hist_align_mask).sum() / hist_align_d

        mu_gt_fut = pred["mu_gt_fut"][:, :active_future_chunks]       # [N, K, z_dim]
        valid_gf = pred["valid_gt_fut"][:, :active_future_chunks]     # [N, K]
        future_z_winner = pred["future_z"][batch_idx, best_mode]  # [N, K, z_dim]
        K_align = min(future_z_winner.size(1), mu_gt_fut.size(1))
        fut_align_mask = valid_gf[:, :K_align].float()
        fut_align_d = fut_align_mask.sum().clamp_min(1.0)
        z_align_fut = ((future_z_winner[:, :K_align] - mu_gt_fut[:, :K_align]).pow(2).sum(-1) * fut_align_mask).sum() / fut_align_d

        z_align_loss = z_align_hist + z_align_fut

        if self.use_aux_loss:
            from model_MoCAR.auxiliary import compute_auxiliary_losses

            aux_losses = compute_auxiliary_losses(
                model=self,
                pred=pred,
                best_mode=best_mode,
                batch_idx=batch_idx,
                hist_target=hist_target,
                hist_target_bridge=hist_target_bridge,
                hist_step_mask=hist_step_mask,
                hist_pair_mask=hist_pair_mask,
                gt_hist_traj=gt_hist_traj,
                hist_traj_mask=hist_traj_mask,
                gt_hist_heading_traj=gt_hist_heading_traj,
                future_target=future_target,
                future_target_bridge=future_target_bridge,
                future_step_mask=future_step_mask,
                future_pair_mask=future_pair_mask,
                gt_future_pos=gt_future_pos,
                gt_future_heading=gt_future_heading,
                reg_mask=reg_mask,
                mu_gt_hist=mu_gt_hist,
                hist_align_mask=hist_align_mask,
                hist_align_d=hist_align_d,
                mu_gt_fut=mu_gt_fut,
                fut_align_mask=fut_align_mask,
                fut_align_d=fut_align_d,
                K_align=K_align,
                bridge_weights=bw,
                aux_last_scale=aux_last_scale,
                aux_decay=aux_decay,
            )
            aux_hist_chunk_loss = aux_losses["aux_hist_chunk_loss"]
            aux_hist_traj_loss = aux_losses["aux_hist_traj_loss"]
            aux_hist_heading_traj_loss = aux_losses["aux_hist_heading_traj_loss"]
            aux_fut_chunk_loss = aux_losses["aux_fut_chunk_loss"]
            aux_fut_traj_wta_loss = aux_losses["aux_fut_traj_wta_loss"]
            aux_fut_heading_traj_loss = aux_losses["aux_fut_heading_traj_loss"]
            aux_z_align_hist = aux_losses["aux_z_align_hist"]
            aux_z_align_fut = aux_losses["aux_z_align_fut"]

        aux_z_align_loss = aux_z_align_hist + aux_z_align_fut

        # ----- Total -----
        total_loss = (
            hist_loss + fut_loss
            + aux_hist_chunk_loss + aux_fut_chunk_loss
            + config["traj_hist_loss_weight"] * hist_traj_loss
            + config["traj_hist_loss_weight"] * aux_hist_traj_loss
            + config["hist_heading_traj_loss_weight"] * hist_heading_traj_loss
            + config["hist_heading_traj_loss_weight"] * aux_hist_heading_traj_loss
            + config["traj_loss_weight"] * traj_wta_loss
            + config["traj_loss_weight"] * aux_fut_traj_wta_loss
            + config["future_heading_traj_loss_weight"] * future_heading_traj_loss
            + config["future_heading_traj_loss_weight"] * aux_fut_heading_traj_loss
            + config["soft_cls_weight"] * soft_cls_loss
            + config["vae_aux_weight"] * vae_loss
            + config["z_align_weight"] * z_align_loss
            + config["z_align_weight"] * aux_z_align_loss
        )

        mode_hist_count = torch.zeros(self.num_modes, device=device)
        mode_hist_count.scatter_add_(0, best_mode, torch.ones_like(best_mode, dtype=torch.float))
        mode_hist_count = mode_hist_count / mode_hist_count.sum().clamp_min(1.0)
        mode_entropy = -(mode_hist_count * (mode_hist_count + 1e-9).log()).sum()

        stats = {
            "loss_total": total_loss.detach(),
            **hist_stats,
            **fut_stats,
            "hist_traj_loss": hist_traj_loss.detach(),
            "traj_wta_loss": traj_wta_loss.detach(),
            "hist_heading_traj_loss": hist_heading_traj_loss.detach(),
            "future_heading_traj_loss": future_heading_traj_loss.detach(),
            "soft_cls_loss": soft_cls_loss.detach(),
            "vae_recon": pred["vae_recon_loss"].detach(),
            "vae_kl": pred["vae_kl_loss"].detach(),
            "z_align": z_align_loss.detach(),
            "z_align_hist": z_align_hist.detach(),
            "z_align_fut": z_align_fut.detach(),
            "aux_hist_chunk_loss": aux_hist_chunk_loss.detach(),
            "aux_hist_traj_loss": aux_hist_traj_loss.detach(),
            "aux_hist_heading_traj_loss": aux_hist_heading_traj_loss.detach(),
            "aux_fut_chunk_loss": aux_fut_chunk_loss.detach(),
            "aux_fut_traj_wta_loss": aux_fut_traj_wta_loss.detach(),
            "aux_fut_heading_traj_loss": aux_fut_heading_traj_loss.detach(),
            "aux_z_align_hist": aux_z_align_hist.detach(),
            "aux_z_align_fut": aux_z_align_fut.detach(),
            "aux_z_align": aux_z_align_loss.detach(),
            "best_mode_mean": best_mode.float().mean().detach(),
            "mode_entropy": mode_entropy.detach(),
            "active_future_chunks": pred["future_rel_pos"].new_tensor(
                float(active_future_chunks)
            ).detach(),
        }

        outputs = {
            "best_mode": best_mode,
            "gt_future_pos": gt_future_pos,
            "gt_future_heading": gt_future_heading,
            "future_valid": future_valid,
        }

        return total_loss, stats, outputs
