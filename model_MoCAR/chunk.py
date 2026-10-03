import os
import math
import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.nn.utils.rnn import pack_padded_sequence

warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

from layers import FourierEmbedding
from layers import MLPLayer
from datamodules import ArgoverseV2DataModule

try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
except Exception:
    wandb = None
    WandbLogger = None


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Invalid boolean value: {v}")


# -------------------------
# Encoder / Decoder (shared)
# -------------------------
class AgentTypeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.type_emb = nn.Embedding(10, hidden_dim)

    def forward(self, agent_type: torch.Tensor) -> torch.Tensor:
        return self.type_emb(agent_type)


class VAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        z_dim: int = 16,
        num_freq_bands: int = 64,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.fourier = FourierEmbedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.agent_type_emb = AgentTypeEmbedding(hidden_dim)
        self.pre_gru_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.to_latent = MLPLayer(hidden_dim, hidden_dim, hidden_dim)
        self.to_mu = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, z_dim))
        self.to_logvar = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, z_dim))

    def forward(
        self,
        step_feat: torch.Tensor,   # [B, P-1, 8]
        trans_mask: torch.Tensor,  # [B, P-1]
        agent_type: torch.Tensor,  # [B]
    ):
        B, T, C = step_feat.shape

        x = self.fourier(step_feat.reshape(B * T, C)).reshape(B, T, self.hidden_dim)
        x = self.pre_gru_norm(x)

        lengths = trans_mask.long().sum(dim=1)      # [B]
        lengths_clamped = lengths.clamp_min(1)

        # reverse in time so the most recent valid transitions are closest to the GRU readout
        x = torch.flip(x, dims=[1])                 # [B, T, H]
        packed = pack_padded_sequence(
            x,
            lengths_clamped.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)                   # [num_layers, B, H]
        pooled = h_n[-1]                            # [B, H]

        type_feat = self.agent_type_emb(agent_type.long())  # [B, H]
        pooled = pooled + type_feat

        zero_len_mask = (lengths == 0)
        if zero_len_mask.any():
            pooled = pooled.clone()
            pooled[zero_len_mask] = 0.0

        latent = self.to_latent(pooled)
        mu = self.to_mu(latent)
        logvar = self.to_logvar(latent)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int = 16,
        hidden_dim: int = 128,
        chunk_points: int = 11,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.chunk_points = chunk_points

        self.time_queries = nn.Parameter(torch.randn(chunk_points, z_dim))
        self.backbone = MLPLayer(z_dim + z_dim, hidden_dim, hidden_dim)
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.delta_head = MLPLayer(hidden_dim, hidden_dim, 2)
        self.vel_head = MLPLayer(hidden_dim, hidden_dim, 2)
        self.head_head = MLPLayer(hidden_dim, hidden_dim, 1)
        self.bridge_head = MLPLayer(z_dim, hidden_dim, 3)

    @staticmethod
    def reconstruct_rel_pos_from_delta_back(delta_back_xy: torch.Tensor) -> torch.Tensor:
        # delta_back_xy: [B, P, 2]
        rev_cum = torch.cumsum(torch.flip(delta_back_xy, dims=[1]), dim=1)
        rel_pos = -torch.flip(rev_cum, dims=[1])
        return rel_pos

    def forward(self, z_q: torch.Tensor):
        B = z_q.size(0)
        tq = self.time_queries.unsqueeze(0).expand(B, -1, -1)  # [B, P, z_dim]
        z_expand = z_q.unsqueeze(1).expand(-1, self.chunk_points, -1)
        feat = self.backbone(torch.cat([z_expand, tq], dim=-1))  # [B, P, H]

        feat_res = feat
        feat, _ = self.temporal_gru(feat)
        feat = self.temporal_norm(feat + feat_res)

        delta_back_xy = self.delta_head(feat)            # [B, P, 2]
        vel = self.vel_head(feat)                        # [B, P, 2]
        heading = self.head_head(feat).squeeze(-1)       # [B, P]
        bridge = self.bridge_head(z_q)                   # [B, 3]
        rel_pos = self.reconstruct_rel_pos_from_delta_back(delta_back_xy)  # [B, P, 2]

        return delta_back_xy, vel, heading, bridge, rel_pos


# -------------------------
# Lightning module
# -------------------------
class VAEModule(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        z_dim: int = 16,
        num_freq_bands: int = 64,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
        chunk_points: int = 11,
        min_valid_points: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        T_max: int = 64,
        kl_w: float = 1e-4,
        kl_warmup_steps: int = 5000,
        bridge_weight_alpha: float = 1.0,
        bridge_weight_angle_ref: float = 0.35,
        bridge_weight_dist_thr: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert chunk_points >= min_valid_points >= 2, (
            f"Require chunk_points >= min_valid_points >= 2, "
            f"but got chunk_points={chunk_points}, min_valid_points={min_valid_points}"
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.kl_w = kl_w
        self.kl_warmup_steps = kl_warmup_steps
        self.bridge_weight_alpha = bridge_weight_alpha
        self.bridge_weight_angle_ref = bridge_weight_angle_ref
        self.bridge_weight_dist_thr = bridge_weight_dist_thr
        self.chunk_points = chunk_points
        self.min_valid_points = min_valid_points

        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            num_freq_bands=num_freq_bands,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
        )
        self.decoder = VAEDecoder(
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            chunk_points=chunk_points,
        )

    # -------- VAE utils --------
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
        if not training:
            return mu
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)

    def kl_beta(self) -> float:
        warm = int(self.kl_warmup_steps)
        if warm <= 0:
            return float(self.kl_w)
        s = min(1.0, float(self.global_step) / float(warm))
        return float(self.kl_w) * s

    def wrap_angle(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

    def coor_global_to_local(self, pos_global, pos_ref, heading_ref, num_nodes):
        cos, sin = heading_ref.cos(), heading_ref.sin()
        rot_mat = torch.zeros(num_nodes, 2, 2, device=pos_global.device, dtype=pos_global.dtype)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        pos_local = torch.bmm(pos_global - pos_ref.unsqueeze(1), rot_mat)
        return pos_local

    def build_chunk(self, data):
        agent = data["agent"]
        valid_mask = agent["valid_mask"]          # [N, T]
        position = agent["position"][..., :2]     # [N, T, 2]
        heading = agent["heading"]                # [N, T]
        velocity = agent["velocity"][..., :2]     # [N, T, 2]
        agent_type = agent["type"]                # [N]
        true_idx = agent["true_idx"]              # [N, 2]
        length = agent["length"]                  # [N]

        device = valid_mask.device
        _, T = valid_mask.shape

        P = self.chunk_points
        S = P - 1
        K = self.min_valid_points

        first_idx = true_idx[:, 0]  # [N]
        last_idx = true_idx[:, 1]   # [N]

        keep_mask = length >= K
        valid_mask = valid_mask[keep_mask]
        position = position[keep_mask]
        heading = heading[keep_mask]
        velocity = velocity[keep_mask]
        agent_type = agent_type[keep_mask]
        first_idx = first_idx[keep_mask]
        last_idx = last_idx[keep_mask]

        M = valid_mask.size(0)

        if M == 0:
            step_feat = position.new_zeros((0, S, 8))
            trans_mask = torch.zeros((0, S), dtype=torch.bool, device=device)
            step_mask = torch.zeros((0, P), dtype=torch.bool, device=device)
            pair_mask = torch.zeros((0,), dtype=torch.float, device=device)
            target_bridge = position.new_zeros((0, 3))
            target = position.new_zeros((0, P, 5))
            return step_feat, trans_mask, step_mask, agent_type, pair_mask, target_bridge, target

        # Need at least K valid points available up to the chosen end_idx
        end_min = first_idx + (K - 1)             # [M]
        num_choices = last_idx - end_min + 1      # [M]

        # Numerical safety; with keep_mask above, this should already be >= 1
        num_choices = num_choices.clamp_min(1)

        p = torch.rand(M, device=device)
        end_idx = end_min + (p * num_choices.float()).long()   # [M]

        rel = torch.arange(P, device=device).unsqueeze(0)       # [1, P]
        chunk_idx = end_idx[:, None] - (P - 1) + rel           # [M, P]
        safe_idx = chunk_idx.clamp(0, T - 1)

        row_idx = torch.arange(M, device=device)[:, None]

        pos_chunk = position[row_idx, safe_idx]    # [M, P, 2]
        head_chunk = heading[row_idx, safe_idx]    # [M, P]
        vel_chunk = velocity[row_idx, safe_idx]    # [M, P, 2]

        # Keep original left-padding / out-of-range masking behavior unchanged
        step_mask = (chunk_idx >= 0) & (chunk_idx < T) & valid_mask[row_idx, safe_idx]  # [M, P]

        pos_ref = pos_chunk[:, -1]    # [M, 2]
        head_ref = head_chunk[:, -1]  # [M]

        pos_local = self.coor_global_to_local(
            pos_chunk, pos_ref, head_ref, num_nodes=M
        )  # [M, P, 2]
        heading_local = self.wrap_angle(head_chunk - head_ref[:, None])  # [M, P]

        dxy = pos_local[:, 1:] - pos_local[:, :-1]  # [M, S, 2]
        dp = torch.norm(dxy, dim=-1)                # [M, S]
        alphap = torch.atan2(dxy[:, :, 1], dxy[:, :, 0])  # [M, S]
        dheading = self.wrap_angle(heading_local[:, 1:] - heading_local[:, :-1])  # [M, S]

        v = torch.norm(vel_chunk, dim=-1)  # [M, P]
        phi = self.wrap_angle(
            torch.atan2(vel_chunk[:, :, 1], vel_chunk[:, :, 0]) - head_ref[:, None]
        )  # [M, P]

        step_feat = torch.cat([
            pos_local[:, 1:],                         # [M, S, 2]
            heading_local[:, 1:].unsqueeze(-1),      # [M, S, 1]
            dp.unsqueeze(-1),                        # [M, S, 1]
            alphap.unsqueeze(-1),                    # [M, S, 1]
            dheading.unsqueeze(-1),                  # [M, S, 1]
            v[:, 1:].unsqueeze(-1),                  # [M, S, 1]
            phi[:, 1:].unsqueeze(-1),                # [M, S, 1]
        ], dim=-1)                                   # [M, S, 8]

        trans_mask = step_mask[:, :-1] & step_mask[:, 1:]  # [M, S]

        step_feat = torch.where(
            trans_mask.unsqueeze(-1),
            step_feat,
            torch.zeros_like(step_feat)
        )

        pair_mask = (step_mask[:, 0] & step_mask[:, -1]).float()  # [M]

        target_bridge = torch.cat([
            pos_local[:, 0],         # [M, 2]
            heading_local[:, 0:1],   # [M, 1]
        ], dim=-1)                   # [M, 3]

        target = torch.cat([
            pos_local,                    # [M, P, 2]
            heading_local.unsqueeze(-1),  # [M, P, 1]
            v.unsqueeze(-1),              # [M, P, 1]
            phi.unsqueeze(-1),            # [M, P, 1]
        ], dim=-1)                        # [M, P, 5]

        return step_feat, trans_mask, step_mask, agent_type, pair_mask, target_bridge, target

    # -------- forward --------
    def forward(self, step_feat, trans_mask, agent_type):
        mu, logvar = self.encoder(step_feat, trans_mask, agent_type)
        z = self.reparameterize(mu, logvar, self.training)
        delta_back_xy, vel, heading, bridge, rel_pos = self.decoder(z)
        return delta_back_xy, vel, heading, bridge, rel_pos, mu, logvar, z

    # -------- loss --------
    def _loss(self, data):
        step_feat, trans_mask, step_mask, agent_type, pair_mask, target_bridge, target = self.build_chunk(data)

        pred_delta_back_xy, pred_vel, pred_heading, pred_bridge, pred_rel_pos, mu, logvar, z = \
            self(step_feat, trans_mask, agent_type)

        if step_feat.size(0) == 0:
            zero = self.encoder.to_mu[-1].weight.sum() * 0.0
            stats = [zero, zero, zero, zero, zero, zero, zero, zero, zero]
            return zero, stats, self.kl_beta()

        # target: [B, P, 5] = [x, y, heading, v, phi]
        target_pos = target[..., 0:2]      # [B, P, 2]
        target_heading = target[..., 2]    # [B, P]
        target_v = target[..., 3]          # [B, P]
        target_phi = target[..., 4]        # [B, P]

        point_mask = step_mask.float()                      # [B, P]
        point_count = point_mask.sum(dim=1).clamp_min(1.0) # [B]

        bridge_xy = target_bridge[:, 0:2]  # [B, 2]
        bridge_h = target_bridge[:, 2]     # [B]
        dist = torch.norm(bridge_xy, dim=-1)  # [B]
        ang = torch.abs(bridge_h)              # [B]

        ang_thr = math.radians(9.0)
        ang_eff = (ang - ang_thr).clamp(min=0.0)
        ang_score = (ang_eff / self.bridge_weight_angle_ref).clamp(max=2.0)
        sample_weight = 1.0 + self.bridge_weight_alpha * ang_score
        sample_weight = torch.where(
            dist < self.bridge_weight_dist_thr,
            torch.ones_like(sample_weight),
            sample_weight
        )  # [B]

        pos_err = (pred_rel_pos - target_pos).pow(2).sum(dim=-1)    # [B, P]
        pos_loss_per = (pos_err * point_mask).sum(dim=1) / point_count

        heading_err = self.wrap_angle(pred_heading - target_heading).pow(2)  # [B, P]
        heading_loss_per = (heading_err * point_mask).sum(dim=1) / point_count

        vx = pred_vel[..., 0]
        vy = pred_vel[..., 1]

        pred_v = torch.sqrt(vx ** 2 + vy ** 2 + 1e-6)
        v_err = (pred_v - target_v).pow(2)
        v_loss_per = (v_err * point_mask).sum(dim=1) / point_count

        pred_phi = torch.atan2(vy, vx)  # [B, P]
        phi_err = self.wrap_angle(pred_phi - target_phi).pow(2)
        phi_mask = point_mask * (target_v > 0.1).float()   # [B, P]
        phi_count = phi_mask.sum(dim=1).clamp_min(1.0)     # [B]
        phi_loss_per = (phi_err * phi_mask).sum(dim=1) / phi_count

        pos_loss = (pos_loss_per * sample_weight).mean()
        heading_loss = (heading_loss_per * sample_weight).mean()
        v_loss = (v_loss_per * sample_weight).mean()
        phi_loss = (phi_loss_per * sample_weight).mean()

        bridge_pos_err = (pred_bridge[:, 0:2] - target_bridge[:, 0:2]).pow(2).sum(dim=-1)  # [B]
        bridge_heading_err = self.wrap_angle(pred_bridge[:, 2] - target_bridge[:, 2]).pow(2)  # [B]

        pair_weight = pair_mask * sample_weight  # [B]
        pair_denom = pair_weight.sum().clamp_min(1.0)

        bridge_pos_loss = (bridge_pos_err * pair_weight).sum() / pair_denom
        bridge_heading_loss = (bridge_heading_err * pair_weight).sum() / pair_denom

        recon = (
            pos_loss
            + heading_loss
            + v_loss
            + phi_loss
            + bridge_pos_loss
            + bridge_heading_loss
        )

        kl_per = self.kl_per_sample(mu, logvar)  # [B]
        kl = kl_per.mean()

        beta = self.kl_beta()
        loss = recon + beta * kl

        stats = [
            recon,
            kl,
            pos_loss,
            heading_loss,
            v_loss,
            phi_loss,
            bridge_pos_loss,
            bridge_heading_loss,
            sample_weight.mean(),
        ]
        return loss, stats, beta

    def training_step(self, data, batch_idx):
        loss, stats, beta = self._loss(data)
        bs = int(getattr(data, "num_graphs", 1))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train_recon", stats[0], on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train_kl", stats[1], on_step=True, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train_beta", beta, on_step=True, on_epoch=False, prog_bar=False, batch_size=bs)

        self.log("train_loss_pos", stats[2], on_step=True, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train_loss_heading", stats[3], on_step=True, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train_loss_v", stats[4], on_step=True, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train_loss_phi", stats[5], on_step=True, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train_loss_bridge_pos", stats[6], on_step=True, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train_loss_bridge_heading", stats[7], on_step=True, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train_sample_weight", stats[8], on_step=True, on_epoch=True, prog_bar=False, batch_size=bs)
        return loss

    def validation_step(self, data, batch_idx):
        loss, stats, beta = self._loss(data)
        bs = int(getattr(data, "num_graphs", 1))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val_recon", stats[0], on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val_kl", stats[1], on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)

        self.log("val_loss_pos", stats[2], on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("val_loss_heading", stats[3], on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("val_loss_v", stats[4], on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("val_loss_phi", stats[5], on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("val_loss_bridge_pos", stats[6], on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("val_loss_bridge_heading", stats[7], on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("val_sample_weight", stats[8], on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(self.T_max), eta_min=0.0)
        return {"optimizer": opt, "lr_scheduler": sch}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VAE_chunk")
        parser.add_argument("--input_dim", type=int, default=8)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--z_dim", type=int, default=24)
        parser.add_argument("--num_freq_bands", type=int, default=64)
        parser.add_argument("--num_gru_layers", type=int, default=1)
        parser.add_argument("--dropout", type=float, default=0.1)

        parser.add_argument("--chunk_points", type=int, default=21)
        parser.add_argument("--min_valid_points", type=int, default=6)

        parser.add_argument("--bridge_weight_alpha", type=float, default=1.0)
        parser.add_argument("--bridge_weight_angle_ref", type=float, default=0.35)
        parser.add_argument("--bridge_weight_dist_thr", type=float, default=0.5)

        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--T_max", type=int, default=500)

        parser.add_argument("--kl_w", type=float, default=1e-4)
        parser.add_argument("--kl_warmup_steps", type=int, default=5000)

        return parent_parser


# -------------------------
# Train entry
# -------------------------
def main():
    pl.seed_everything(2026, workers=True)

    parser = ArgumentParser()

    # ===== dataset/datamodule =====
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--processed", type=str, default="data")
    parser.add_argument("--train_batch_size", type=int, default=200)
    parser.add_argument("--val_batch_size", type=int, default=200)
    parser.add_argument("--test_batch_size", type=int, default=200)
    parser.add_argument("--shuffle", type=str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=str2bool, default=True)

    # ===== trainer =====
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--log_every_n_steps", type=int, default=200)

    # ===== logging / resume =====
    parser.add_argument("--wandb_project", type=str, default="VAE_CHUNK_21")
    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--resume_ckpt", type=str, default="./VAE_CHUNK_21/7wqk63ir/checkpoints/last.ckpt")

    VAEModule.add_model_specific_args(parser)
    args = parser.parse_args()

    model = VAEModule(**vars(args))
    datamodule = ArgoverseV2DataModule(**vars(args))

    ckpt = ModelCheckpoint(
        monitor="val_recon",
        save_top_k=8,
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = None
    if args.use_wandb and WandbLogger is not None:
        logger = WandbLogger(project=args.wandb_project)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="auto",
        precision=args.precision,
        callbacks=[ckpt, lr_monitor],
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
    )

    ckpt_path = args.resume_ckpt if (args.resume_ckpt and os.path.isfile(args.resume_ckpt)) else None
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    if wandb is not None and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
