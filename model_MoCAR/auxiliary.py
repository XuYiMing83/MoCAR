import torch

from utils import wrap_angle


def compute_auxiliary_losses(
    model,
    pred,
    best_mode,
    batch_idx,
    hist_target,
    hist_target_bridge,
    hist_step_mask,
    hist_pair_mask,
    gt_hist_traj,
    hist_traj_mask,
    gt_hist_heading_traj,
    future_target,
    future_target_bridge,
    future_step_mask,
    future_pair_mask,
    gt_future_pos,
    gt_future_heading,
    reg_mask,
    mu_gt_hist,
    hist_align_mask,
    hist_align_d,
    mu_gt_fut,
    fut_align_mask,
    fut_align_d,
    K_align,
    bridge_weights,
    aux_last_scale,
    aux_decay,
):
    device = pred["x_mode"].device
    zero = torch.tensor(0.0, device=device)
    losses = {
        "aux_hist_chunk_loss": zero,
        "aux_hist_traj_loss": zero,
        "aux_hist_heading_traj_loss": zero,
        "aux_fut_chunk_loss": zero,
        "aux_fut_traj_wta_loss": zero,
        "aux_fut_heading_traj_loss": zero,
        "aux_z_align_hist": zero,
        "aux_z_align_fut": zero,
    }

    num_aux_layers = int(pred["aux_hist_rel_pos"].size(0)) if pred["aux_hist_rel_pos"].numel() > 0 else 0
    if num_aux_layers == 0:
        return losses

    aux_layer_w = torch.tensor(
        [aux_last_scale * (aux_decay ** (num_aux_layers - 1 - layer)) for layer in range(num_aux_layers)],
        dtype=torch.float,
        device=device,
    )

    aux_hist_chunk_losses = []
    aux_hist_traj_losses = []
    aux_hist_heading_losses = []
    aux_hist_z_losses = []
    aux_fut_chunk_losses = []
    aux_fut_traj_losses = []
    aux_fut_heading_losses = []
    aux_fut_z_losses = []

    for layer in range(num_aux_layers):
        hist_chunk_loss, _ = model.chunk_recon_loss(
            pred["aux_hist_rel_pos"][layer],
            pred["aux_hist_heading"][layer],
            pred["aux_hist_vel"][layer],
            pred["aux_hist_bridge"][layer],
            hist_target,
            hist_target_bridge,
            hist_step_mask,
            hist_pair_mask,
            **bridge_weights,
        )
        aux_hist_chunk_losses.append(hist_chunk_loss)

        hist_traj_l1 = torch.abs(pred["aux_hist_traj"][layer] - gt_hist_traj).sum(dim=-1)
        hist_traj_loss = (hist_traj_l1 * hist_traj_mask).sum(0) / hist_traj_mask.sum(0).clamp_min(1.0)
        aux_hist_traj_losses.append(hist_traj_loss.mean())

        hist_heading_err = wrap_angle(pred["aux_hist_heading_traj"][layer] - gt_hist_heading_traj).abs()
        hist_heading_loss = (hist_heading_err * hist_traj_mask).sum() / hist_traj_mask.sum().clamp_min(1.0)
        aux_hist_heading_losses.append(hist_heading_loss)

        hist_z_loss = ((pred["aux_hist_z"][layer] - mu_gt_hist).pow(2).sum(-1) * hist_align_mask).sum() / hist_align_d
        aux_hist_z_losses.append(hist_z_loss)

        fut_chunk_loss, _ = model.chunk_recon_loss(
            pred["aux_future_rel_pos"][layer][batch_idx, best_mode],
            pred["aux_future_heading"][layer][batch_idx, best_mode],
            pred["aux_future_vel"][layer][batch_idx, best_mode],
            pred["aux_future_bridge"][layer][batch_idx, best_mode],
            future_target,
            future_target_bridge,
            future_step_mask,
            future_pair_mask,
            **bridge_weights,
        )
        aux_fut_chunk_losses.append(fut_chunk_loss)

        fut_traj_best = pred["aux_future_traj"][layer][batch_idx, best_mode]
        fut_traj_l1 = torch.abs(fut_traj_best - gt_future_pos).sum(dim=-1)
        fut_traj_loss = (fut_traj_l1 * reg_mask).sum(0) / reg_mask.sum(0).clamp_min(1.0)
        aux_fut_traj_losses.append(fut_traj_loss.mean())

        fut_heading_best = pred["aux_future_heading_traj"][layer][batch_idx, best_mode]
        fut_heading_err = wrap_angle(fut_heading_best - gt_future_heading).abs()
        fut_heading_loss = (fut_heading_err * reg_mask).sum() / reg_mask.sum().clamp_min(1.0)
        aux_fut_heading_losses.append(fut_heading_loss)

        fut_z_winner = pred["aux_future_z"][layer][batch_idx, best_mode]
        fut_z_loss = (
            ((fut_z_winner[:, :K_align] - mu_gt_fut[:, :K_align]).pow(2).sum(-1) * fut_align_mask).sum()
            / fut_align_d
        )
        aux_fut_z_losses.append(fut_z_loss)

    losses["aux_hist_chunk_loss"] = (torch.stack(aux_hist_chunk_losses) * aux_layer_w).sum()
    losses["aux_hist_traj_loss"] = (torch.stack(aux_hist_traj_losses) * aux_layer_w).sum()
    losses["aux_hist_heading_traj_loss"] = (torch.stack(aux_hist_heading_losses) * aux_layer_w).sum()
    losses["aux_z_align_hist"] = (torch.stack(aux_hist_z_losses) * aux_layer_w).sum()
    losses["aux_fut_chunk_loss"] = (torch.stack(aux_fut_chunk_losses) * aux_layer_w).sum()
    losses["aux_fut_traj_wta_loss"] = (torch.stack(aux_fut_traj_losses) * aux_layer_w).sum()
    losses["aux_fut_heading_traj_loss"] = (torch.stack(aux_fut_heading_losses) * aux_layer_w).sum()
    losses["aux_z_align_fut"] = (torch.stack(aux_fut_z_losses) * aux_layer_w).sum()
    return losses
