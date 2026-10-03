import math

import torch

from utils import wrap_angle


class ChunkReconstructionLoss:
    """Loss for decoded endpoint-normalized trajectory tokens."""

    def chunk_recon_loss(
        self,
        pred_rel_pos,
        pred_heading,
        pred_vel,
        pred_bridge,
        target,
        target_bridge,
        step_mask,
        pair_mask,
        bridge_weight_alpha=1.0,
        bridge_weight_angle_ref=0.35,
        bridge_weight_dist_thr=0.5,
    ):
        device = pred_rel_pos.device
        N, C, P = pred_heading.shape
        B = N * C

        pred_rel_pos = pred_rel_pos.reshape(B, P, 2)
        pred_heading = pred_heading.reshape(B, P)
        pred_vel = pred_vel.reshape(B, P, 2)
        pred_bridge = pred_bridge.reshape(B, 3)
        target = target.reshape(B, P, 5)
        target_bridge = target_bridge.reshape(B, 3)
        step_mask = step_mask.reshape(B, P).float()
        pair_mask = pair_mask.reshape(B).float()

        target_pos = target[..., 0:2]
        target_heading = target[..., 2]
        target_v = target[..., 3]
        target_phi = target[..., 4]

        bridge_xy = target_bridge[:, 0:2]
        bridge_h = target_bridge[:, 2]
        dist = torch.norm(bridge_xy, dim=-1)
        ang = torch.abs(bridge_h)

        ang_thr = torch.tensor(math.radians(9.0), device=device)
        ang_eff = (ang - ang_thr).clamp(min=0.0)
        ang_score = (ang_eff / bridge_weight_angle_ref).clamp(max=2.0)
        sample_weight = 1.0 + bridge_weight_alpha * ang_score
        sample_weight = torch.where(dist < bridge_weight_dist_thr, torch.ones_like(sample_weight), sample_weight)

        point_count = step_mask.sum(dim=1).clamp_min(1.0)

        pos_loss = ((pred_rel_pos - target_pos).pow(2).sum(-1) * step_mask).sum(1) / point_count
        head_loss = (wrap_angle(pred_heading - target_heading).pow(2) * step_mask).sum(1) / point_count

        vx, vy = pred_vel[..., 0], pred_vel[..., 1]
        pred_v = torch.sqrt(vx**2 + vy**2 + 1e-6)
        v_loss = ((pred_v - target_v).pow(2) * step_mask).sum(1) / point_count

        pred_phi = torch.atan2(vy, vx)
        phi_mask = step_mask * (target_v > 0.1).float()
        phi_count = phi_mask.sum(1).clamp_min(1.0)
        phi_loss = (wrap_angle(pred_phi - target_phi).pow(2) * phi_mask).sum(1) / phi_count

        pos_l = (pos_loss * sample_weight).mean()
        head_l = (head_loss * sample_weight).mean()
        v_l = (v_loss * sample_weight).mean()
        phi_l = (phi_loss * sample_weight).mean()

        pair_w = pair_mask * sample_weight
        pair_d = pair_w.sum().clamp_min(1.0)
        bridge_pos_l = ((pred_bridge[:, :2] - target_bridge[:, :2]).pow(2).sum(-1) * pair_w).sum() / pair_d
        bridge_head_l = (wrap_angle(pred_bridge[:, 2] - target_bridge[:, 2]).pow(2) * pair_w).sum() / pair_d

        total = pos_l + head_l + v_l + phi_l + bridge_pos_l + bridge_head_l
        stats = {
            "pos": pos_l,
            "heading": head_l,
            "v": v_l,
            "phi": phi_l,
            "bridge_pos": bridge_pos_l,
            "bridge_head": bridge_head_l,
            "total": total,
        }
        return total, {k: v.detach() for k, v in stats.items()}
