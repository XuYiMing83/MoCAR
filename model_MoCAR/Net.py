import math
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData

from metrics import Brier, MR, minADE, minAHE, minFDE, minFHE
from model_MoCAR.map_encoder import MapEncoder
from model_MoCAR.decoder import ChunkAgent
from utils import wrap_angle

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class Net(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 2,
        z_dim: int = 24,
        hidden_dim: int = 128,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        time_span: int = 10,
        pl2pl_radius: float = 150,
        a2a_radius: float = 50,
        pl2a_radius: float = 50,
        num_freq_bands: int = 64,
        num_map_layers: int = 1,
        num_heads: int = 8,
        head_dim: int = 16,
        dropout: float = 0.1,
        num_modes: int = 6,
        num_dec_layers: int = 5,
        num_future_chunks: Optional[int] = None,
        chunk_points: int = 16,
        min_valid_points: int = 6,
        chunk_ckpt_path: Optional[str] = None,
        use_aux_loss: bool = False,
        chunk_noise_ratio: float = 0.0,
        chunk_noise_sigma: float = 0.0,
        rollout_warmup_epochs: int = 3,
        rollout_warmup_chunks: int = 1,
        # loss weights
        traj_hist_loss_weight: float = 1,
        traj_loss_weight: float = 10,
        soft_cls_weight: float = 0.2,
        soft_cls_temp: float = 1.0,
        bridge_weight_alpha: float = 1.0,
        bridge_weight_angle_ref: float = 0.35,
        bridge_weight_dist_thr: float = 0.5,
        # [NEW] VAE joint training weights
        vae_aux_weight: float = 0.1,
        z_align_weight: float = 0.1,
        vae_kl_weight: float = 1e-4,
        vae_lr_scale: float = 0.1,              # [NEW] VAE lr = base_lr * vae_lr_scale
        # optim
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        T_max: int = 64,
        warmup_epochs: int = 5,
        **kwargs,
    ):
        super().__init__()
        if rollout_warmup_epochs < 0:
            raise ValueError("rollout_warmup_epochs must be non-negative")
        if rollout_warmup_chunks < 1:
            raise ValueError("rollout_warmup_chunks must be at least 1")
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span
        self.pl2pl_radius = pl2pl_radius
        self.a2a_radius = a2a_radius
        self.pl2a_radius = pl2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_modes = num_modes
        self.num_dec_layers = num_dec_layers
        self.chunk_points = chunk_points
        self.chunk_stride = chunk_points - 1
        self.min_valid_points = min_valid_points
        self.rollout_warmup_epochs = rollout_warmup_epochs
        self.rollout_warmup_chunks = rollout_warmup_chunks
        assert self.chunk_points >= self.min_valid_points >= 2
        if num_future_chunks is None:
            self.num_future_chunks = math.ceil(num_future_steps / self.chunk_stride)
        else:
            self.num_future_chunks = num_future_chunks
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs

        self.traj_hist_loss_weight = traj_hist_loss_weight
        self.traj_loss_weight = traj_loss_weight
        self.soft_cls_weight = soft_cls_weight
        self.soft_cls_temp = soft_cls_temp
        self.bridge_weight_alpha = bridge_weight_alpha
        self.bridge_weight_angle_ref = bridge_weight_angle_ref
        self.bridge_weight_dist_thr = bridge_weight_dist_thr

        # [NEW] VAE joint training params
        self.vae_aux_weight = vae_aux_weight
        self.z_align_weight = z_align_weight
        self.vae_kl_weight = vae_kl_weight
        self.vae_lr_scale = vae_lr_scale

        chunk_anchor = self._build_chunk_anchor(num_historical_steps, self.chunk_stride)
        self.register_buffer("chunk_anchor", chunk_anchor, persistent=False)

        self.map_encoder = MapEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = ChunkAgent(
            input_dim=input_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_modes=num_modes,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            num_future_chunks=self.num_future_chunks,
            chunk_points=self.chunk_points,
            min_valid_points=self.min_valid_points,
            chunk_ckpt_path=chunk_ckpt_path,
            use_aux_loss=use_aux_loss,
            chunk_noise_ratio=chunk_noise_ratio,
            chunk_noise_sigma=chunk_noise_sigma,
        )

        # metrics
        self.Brier = Brier(max_guesses=num_modes)
        self.minADE = minADE(max_guesses=num_modes)
        self.minADE_1 = minADE(max_guesses=1)
        self.minAHE = minAHE(max_guesses=num_modes)
        self.minFDE = minFDE(max_guesses=num_modes)
        self.minFDE_1 = minFDE(max_guesses=1)
        self.minFHE = minFHE(max_guesses=num_modes)
        self.MR = MR(max_guesses=num_modes)

    @staticmethod
    def _build_chunk_anchor(num_historical_steps: int, chunk_stride: int) -> torch.Tensor:
        last_idx = num_historical_steps - 1
        anchors = []
        cur = last_idx
        while cur >= 0:
            anchors.append(cur)
            cur -= chunk_stride
        anchors = anchors[::-1]
        return torch.tensor(anchors, dtype=torch.long)

    @staticmethod
    def coor_global_to_local(
        pos_global: torch.Tensor,
        pos_ref: torch.Tensor,
        heading_ref: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        cos, sin = heading_ref.cos(), heading_ref.sin()
        rot_mat = torch.zeros(
            num_nodes, 2, 2,
            device=pos_global.device, dtype=pos_global.dtype,
        )
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        return torch.bmm(pos_global - pos_ref.unsqueeze(1), rot_mat)

    def forward(self, data: HeteroData, active_future_chunks: Optional[int] = None):
        map_enc = self.map_encoder(data)
        pred = self.decoder(data, map_enc, active_future_chunks=active_future_chunks)
        return pred

    def current_active_future_chunks(self) -> Optional[int]:
        if self.rollout_warmup_epochs > 0 and self.current_epoch < self.rollout_warmup_epochs:
            return self.rollout_warmup_chunks
        return None

    def calculate_loss(self, pred, data, return_outputs=False):
        config = {
            "traj_hist_loss_weight": self.traj_hist_loss_weight,
            "traj_loss_weight": self.traj_loss_weight,
            "soft_cls_weight": self.soft_cls_weight,
            "soft_cls_temp": self.soft_cls_temp,
            "vae_aux_weight": self.vae_aux_weight,
            "vae_kl_weight": self.vae_kl_weight,
            "z_align_weight": self.z_align_weight,
            "bridge_weight_alpha": self.bridge_weight_alpha,
            "bridge_weight_angle_ref": self.bridge_weight_angle_ref,
            "bridge_weight_dist_thr": self.bridge_weight_dist_thr,

            "aux_last_layer_scale": 0.5,
            "aux_layer_decay": 0.5,

            "hist_heading_traj_loss_weight": 1,
            "future_heading_traj_loss_weight": 1,
        }
        total_loss, stats, outputs = self.decoder.compute_all_losses(pred, data, config)
        if return_outputs:
            return total_loss, stats, outputs
        return total_loss, stats

    def training_step(self, data, batch_idx):
        pred = self(data, active_future_chunks=self.current_active_future_chunks())
        loss, stats = self.calculate_loss(pred, data, return_outputs=False)
        bs = int(getattr(data, "num_graphs", 1))

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        for k, v in stats.items():
            self.log(f"train_{k}", v, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        return loss

    def validation_step(self, data, batch_idx):
        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

        pred = self(data)
        loss, stats, outputs = self.calculate_loss(pred, data, return_outputs=True)
        bs = int(getattr(data, "num_graphs", 1))

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        for k, v in stats.items():
            self.log(f"val_{k}", v, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)

        # validation metrics
        future_valid = outputs["future_valid"]
        gt_future_pos = outputs["gt_future_pos"]
        gt_future_heading = outputs["gt_future_heading"]
        eval_mask = data["agent"]["category"] == 3

        valid_mask_eval = future_valid[eval_mask]

        traj_eval = torch.cat([
            pred["future_traj"][eval_mask],
            pred["future_heading_traj"][eval_mask].unsqueeze(-1),
        ], dim=-1)

        pi_eval = pred["mode_prob"][eval_mask]

        gt_eval = torch.cat([
            gt_future_pos[eval_mask],
            gt_future_heading[eval_mask].unsqueeze(-1),
        ], dim=-1)

        self.Brier.update(pred=traj_eval[..., :self.input_dim], target=gt_eval[..., :self.input_dim], prob=pi_eval, valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.input_dim], target=gt_eval[..., :self.input_dim], prob=pi_eval, valid_mask=valid_mask_eval)
        self.minADE_1.update(pred=traj_eval[..., :self.input_dim], target=gt_eval[..., :self.input_dim], prob=pi_eval, valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.input_dim], target=gt_eval[..., :self.input_dim], prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE_1.update(pred=traj_eval[..., :self.input_dim], target=gt_eval[..., :self.input_dim], prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.input_dim], target=gt_eval[..., :self.input_dim], prob=pi_eval, valid_mask=valid_mask_eval)

        eval_bs = int(gt_eval.size(0))
        self.log("val_Brier", self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=eval_bs, sync_dist=True)
        self.log("val_minADE", self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=eval_bs, sync_dist=True)
        self.log("val_minADE_1", self.minADE_1, prog_bar=True, on_step=False, on_epoch=True, batch_size=eval_bs, sync_dist=True)
        self.log("val_minAHE", self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=eval_bs, sync_dist=True)
        self.log("val_minFDE", self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=eval_bs, sync_dist=True)
        self.log("val_minFDE_1", self.minFDE_1, prog_bar=True, on_step=False, on_epoch=True, batch_size=eval_bs, sync_dist=True)
        self.log("val_minFHE", self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=eval_bs, sync_dist=True)
        self.log("val_MR", self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=eval_bs, sync_dist=True)

        return {"loss": loss, **stats, **outputs}

    def on_after_backward(self):
        nan_params = []
        for name, p in self.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                nan_params.append((name, (~torch.isfinite(p.grad)).sum().item()))
        if nan_params:
            print(f"[NaN grad] {len(nan_params)} params affected:")
            for name, count in nan_params[:10]:
                print(f"  {name}: {count} NaN elements")
            raise RuntimeError("NaN gradient detected")

    # ==================================================================
    # [CHANGED] configure_optimizers - 3 param groups with different lr
    # ==================================================================
    def configure_optimizers(self):
        # Collect VAE encoder/decoder param names
        vae_param_names = set()
        for name, _ in self.decoder.chunk_encoder.named_parameters():
            vae_param_names.add(f"decoder.chunk_encoder.{name}")
        for name, _ in self.decoder.chunk_decoder.named_parameters():
            vae_param_names.add(f"decoder.chunk_decoder.{name}")

        # Standard decay/no_decay classification
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.MultiheadAttention, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.LayerNorm, nn.Embedding
        )
        trainable_param_names = {
            name for name, p in self.named_parameters() if p.requires_grad
        }
        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                if full_param_name not in trainable_param_names:
                    continue
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                    else:
                        no_decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)

        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both decay/no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters missing: {param_dict.keys() - union_params}"

        vae_lr = self.lr * self.vae_lr_scale

        main_decay_params = sorted([n for n in decay if n not in vae_param_names])
        main_no_decay_params = sorted([n for n in no_decay if n not in vae_param_names])
        vae_params = sorted([n for n in union_params if n in vae_param_names])

        print(f"[Optimizer] main_decay: {len(main_decay_params)} params, lr={self.lr}")
        print(f"[Optimizer] main_no_decay: {len(main_no_decay_params)} params, lr={self.lr}")
        print(f"[Optimizer] vae: {len(vae_params)} params, lr={vae_lr}")

        optim_groups = [
            {
                "params": [param_dict[n] for n in main_decay_params],
                "weight_decay": self.weight_decay,
                "lr": self.lr,
            },
            {
                "params": [param_dict[n] for n in main_no_decay_params],
                "weight_decay": 0.0,
                "lr": self.lr,
            },
            {
                "params": [param_dict[n] for n in vae_params],
                "weight_decay": 0.0,
                "lr": vae_lr,    # [NEW] VAE gets lower lr
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-6,
        )

        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.2, end_factor=1.0, total_iters=self.warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=max(1, self.T_max - self.warmup_epochs), eta_min=0.0,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.T_max, eta_min=0.0,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ChunkNet")
        parser.add_argument("--input_dim", type=int, default=2)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--z_dim", type=int, default=24)
        parser.add_argument("--num_historical_steps", type=int, default=50)
        parser.add_argument("--num_future_steps", type=int, default=60)
        parser.add_argument("--num_modes", type=int, default=6)
        parser.add_argument("--num_freq_bands", type=int, default=64)
        parser.add_argument("--num_map_layers", type=int, default=1)
        parser.add_argument("--num_dec_layers", type=int, default=5)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--head_dim", type=int, default=16)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--pl2pl_radius", type=float, default=150)
        parser.add_argument("--pl2a_radius", type=float, default=50)
        parser.add_argument("--a2a_radius", type=float, default=50)
        parser.add_argument("--chunk_points", type=int, default=16)
        parser.add_argument("--min_valid_points", type=int, default=6)
        parser.add_argument("--num_future_chunks", type=int, default=None)

        parser.add_argument("--chunk_ckpt_path", type=str, default=None)
        parser.add_argument("--use_aux_loss", action="store_true")
        parser.add_argument("--chunk_noise_ratio", type=float, default=0.0)
        parser.add_argument("--chunk_noise_sigma", type=float, default=0.0)
        parser.add_argument("--rollout_warmup_epochs", type=int, default=3)
        parser.add_argument("--rollout_warmup_chunks", type=int, default=1)

        parser.add_argument("--traj_hist_loss_weight", type=float, default=1.0)
        parser.add_argument("--traj_loss_weight", type=float, default=20.0)
        parser.add_argument("--soft_cls_weight", type=float, default=1.0)
        parser.add_argument("--soft_cls_temp", type=float, default=1.0)

        parser.add_argument("--bridge_weight_alpha", type=float, default=1.0)
        parser.add_argument("--bridge_weight_angle_ref", type=float, default=0.35)
        parser.add_argument("--bridge_weight_dist_thr", type=float, default=0.5)

        # [NEW] VAE joint training args
        parser.add_argument("--vae_aux_weight", type=float, default=0.1)
        parser.add_argument("--z_align_weight", type=float, default=0.01)
        parser.add_argument("--vae_kl_weight", type=float, default=1e-4)
        parser.add_argument("--vae_lr_scale", type=float, default=0.1)

        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--T_max", type=int, default=64)
        parser.add_argument("--warmup_epochs", type=int, default=0)
        return parent_parser
