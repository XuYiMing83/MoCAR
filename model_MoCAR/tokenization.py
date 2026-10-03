from typing import Dict

import torch

from utils import wrap_angle


class EndAlignedTrajectoryTokenizer:
    """Build trajectory tokens in the coordinate frame of each chunk endpoint."""

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
    def repair_valid_mask(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.clone().bool()
        rise = (~mask[:, :-1]) & mask[:, 1:]
        mask[:, :-1] |= rise
        return mask

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
            device=pos_global.device,
            dtype=pos_global.dtype,
        )
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        return torch.bmm(pos_global - pos_ref.unsqueeze(1), rot_mat)

    @staticmethod
    def local_vec_to_global(
        vec_local: torch.Tensor,
        heading_ref: torch.Tensor,
    ) -> torch.Tensor:
        while heading_ref.dim() < vec_local.dim() - 1:
            heading_ref = heading_ref.unsqueeze(-1)
        c = torch.cos(heading_ref)
        s = torch.sin(heading_ref)
        x = vec_local[..., 0]
        y = vec_local[..., 1]
        return torch.stack([x * c - y * s, x * s + y * c], dim=-1)

    def augment_chunk_positions(self, pos_a: torch.Tensor, agent_type: torch.Tensor) -> torch.Tensor:
        ratio = float(getattr(self, "chunk_noise_ratio", 0.0))
        sigma = float(getattr(self, "chunk_noise_sigma", 0.0))
        if (not getattr(self, "training", False)) or ratio <= 0.0 or sigma <= 0.0 or pos_a.numel() == 0:
            return pos_a

        ratio = min(max(ratio, 0.0), 1.0)
        vehicle_types = torch.tensor([0, 4, 5, 6, 7], device=agent_type.device, dtype=agent_type.dtype)
        vehicle_mask = (agent_type[:, None] == vehicle_types[None, :]).any(dim=-1)
        agent_mask = (torch.rand(pos_a.size(0), device=pos_a.device) < ratio) & vehicle_mask
        if not agent_mask.any():
            return pos_a

        noise = torch.randn_like(pos_a) * sigma
        return torch.where(agent_mask[:, None, None], pos_a + noise, pos_a)

    def build_chunk_bundle_from_anchors(
        self,
        pos_a: torch.Tensor,
        head_a: torch.Tensor,
        vel_a: torch.Tensor,
        mask: torch.Tensor,
        agent_type: torch.Tensor,
        anchor_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = pos_a.device
        pos_a = self.augment_chunk_positions(pos_a=pos_a, agent_type=agent_type)
        num_nodes, T, _ = pos_a.shape
        num_chunks = int(anchor_idx.numel())

        P = self.chunk_points
        S = self.chunk_stride
        K = self.min_valid_points

        rel = torch.arange(P, device=device).view(1, 1, P)
        chunk_idx = anchor_idx.view(1, num_chunks, 1) - S + rel
        safe_idx = chunk_idx.clamp(min=0, max=T - 1).expand(num_nodes, -1, -1)

        row_idx = torch.arange(num_nodes, device=device)[:, None, None]

        pos_chunk = pos_a[row_idx, safe_idx]
        head_chunk = head_a[row_idx, safe_idx]
        vel_chunk = vel_a[row_idx, safe_idx]
        valid_chunk = mask[row_idx, safe_idx]

        in_range = (chunk_idx >= 0) & (chunk_idx < T)
        step_mask = valid_chunk & in_range

        safe_anchor = anchor_idx.clamp(min=0, max=T - 1)
        anchor_valid = mask[:, safe_anchor]
        chunk_valid = (step_mask.sum(dim=-1) >= K) & anchor_valid
        step_mask = step_mask & chunk_valid.unsqueeze(-1)
        pair_mask = step_mask[..., 0] & step_mask[..., -1]

        B = num_nodes * num_chunks
        pos_flat = pos_chunk.reshape(B, P, 2)
        head_flat = head_chunk.reshape(B, P)
        vel_flat = vel_chunk.reshape(B, P, 2)
        step_mask_flat = step_mask.reshape(B, P)
        chunk_valid_flat = chunk_valid.reshape(B)
        agent_type_chunk = agent_type[:, None].expand(-1, num_chunks).reshape(-1)

        pos_ref = pos_flat[:, -1]
        head_ref = head_flat[:, -1]
        pos_local = self.coor_global_to_local(pos_flat, pos_ref, head_ref, B)
        heading_local = wrap_angle(head_flat - head_ref[:, None])

        dxy = pos_local[:, 1:] - pos_local[:, :-1]
        dp = torch.norm(dxy, dim=-1)
        alphap = torch.atan2(dxy[..., 1], dxy[..., 0])
        dheading = wrap_angle(heading_local[:, 1:] - heading_local[:, :-1])

        v = torch.norm(vel_flat, dim=-1)
        phi = wrap_angle(torch.atan2(vel_flat[..., 1], vel_flat[..., 0]) - head_ref[:, None])

        step_feat = torch.cat([
            pos_local[:, 1:],
            heading_local[:, 1:].unsqueeze(-1),
            dp.unsqueeze(-1),
            alphap.unsqueeze(-1),
            dheading.unsqueeze(-1),
            v[:, 1:].unsqueeze(-1),
            phi[:, 1:].unsqueeze(-1),
        ], dim=-1)

        trans_mask = step_mask_flat[:, :-1] & step_mask_flat[:, 1:]
        step_feat = torch.where(trans_mask.unsqueeze(-1), step_feat, torch.zeros_like(step_feat))

        pos_local_4d = pos_local.view(num_nodes, num_chunks, P, 2)
        heading_local_3d = heading_local.view(num_nodes, num_chunks, P)
        v_3d = v.view(num_nodes, num_chunks, P)
        phi_3d = phi.view(num_nodes, num_chunks, P)

        target = torch.cat([
            pos_local_4d,
            heading_local_3d.unsqueeze(-1),
            v_3d.unsqueeze(-1),
            phi_3d.unsqueeze(-1),
        ], dim=-1)
        target_bridge = torch.cat([pos_local_4d[:, :, 0], heading_local_3d[:, :, 0:1]], dim=-1)

        return {
            "step_feat": step_feat,
            "trans_mask": trans_mask,
            "chunk_valid": chunk_valid,
            "chunk_valid_flat": chunk_valid_flat,
            "agent_type_chunk": agent_type_chunk,
            "target": target,
            "target_bridge": target_bridge,
            "step_mask": step_mask,
            "pair_mask": pair_mask,
        }
