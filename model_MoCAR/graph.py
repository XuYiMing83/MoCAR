import torch


class TokenGraphBuilder:
    """Low-level sparse graph construction used by the token decoder."""

    @staticmethod
    def build_self_loop_edges(valid_flat: torch.Tensor) -> torch.Tensor:
        device = valid_flat.device
        idx = valid_flat.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        return torch.stack([idx, idx], dim=0)

    @staticmethod
    def build_same_agent_bipartite_edges(
        src_valid: torch.Tensor,
        dst_valid: torch.Tensor,
    ) -> torch.Tensor:
        device = src_valid.device
        N, S = src_valid.shape
        D = dst_valid.size(1)
        edge_mask = src_valid[:, :, None] & dst_valid[:, None, :]
        if not edge_mask.any():
            return torch.empty(2, 0, dtype=torch.long, device=device)
        agent_id = torch.arange(N, device=device)[:, None, None]
        s_id = torch.arange(S, device=device)[None, :, None]
        d_id = torch.arange(D, device=device)[None, None, :]
        src = (agent_id * S + s_id).expand(N, S, D)
        dst = (agent_id * D + d_id).expand(N, S, D)
        return torch.stack([src[edge_mask], dst[edge_mask]], dim=0)

    @staticmethod
    def build_same_agent_dense_edges(valid: torch.Tensor, loop: bool = False) -> torch.Tensor:
        device = valid.device
        N, K = valid.shape
        src_local = torch.arange(K, device=device)[:, None].expand(K, K)
        dst_local = torch.arange(K, device=device)[None, :].expand(K, K)
        if not loop:
            keep = src_local != dst_local
            src_local = src_local[keep]
            dst_local = dst_local[keep]
        else:
            src_local = src_local.reshape(-1)
            dst_local = dst_local.reshape(-1)
        if src_local.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        src_local = src_local.view(1, -1).expand(N, -1)
        dst_local = dst_local.view(1, -1).expand(N, -1)
        edge_mask = valid.gather(1, src_local) & valid.gather(1, dst_local)
        if not edge_mask.any():
            return torch.empty(2, 0, dtype=torch.long, device=device)
        agent_id = torch.arange(N, device=device)[:, None]
        src = agent_id * K + src_local
        dst = agent_id * K + dst_local
        return torch.stack([src[edge_mask], dst[edge_mask]], dim=0)

    @staticmethod
    def build_same_agent_mode_bipartite_edges(
        src_valid: torch.Tensor,
        dst_valid: torch.Tensor,
    ) -> torch.Tensor:
        device = src_valid.device
        N, M, S = src_valid.shape
        edge_mask = src_valid & dst_valid.unsqueeze(-1)
        if not edge_mask.any():
            return torch.empty(2, 0, dtype=torch.long, device=device)
        agent_id = torch.arange(N, device=device)[:, None, None]
        mode_id = torch.arange(M, device=device)[None, :, None]
        step_id = torch.arange(S, device=device)[None, None, :]
        src = (((agent_id * M) + mode_id) * S + step_id).expand(N, M, S)
        dst = ((agent_id * M) + mode_id).expand(N, M, S)
        return torch.stack([src[edge_mask], dst[edge_mask]], dim=0)
