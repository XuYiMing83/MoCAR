from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import transforms
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, Polygon, Rectangle
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Datasets.dataset_pkl import AVDataset
from model_MoCAR.checkpoint import load_weights
from model_MoCAR.Net import Net
from transforms import TargetBuilder

try:
    from av2.map.map_api import ArgoverseStaticMap
except ImportError:
    ArgoverseStaticMap = None

warnings.filterwarnings("ignore", category=UserWarning)


# PyCharm-friendly defaults. Command-line arguments override these values.
ROOT = "/mnt/d/av2_data"
PROCESSED = "MoCAR_data"
SPLIT = "val"
MAP_ROOT = "/mnt/d/av2/val"
CHECKPOINT = "./checkpoint/MoCAR_weights.ckpt"
OUTPUT_DIR = "./visualize"
START_INDEX = 14500
END_INDEX = 15500
CATEGORIES = (2, 3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class VisualizationStyle:
    background: str = "#F7F8FC"
    lane_face: str = "#D6DBE6"
    lane_edge: str = "#707A8C"
    crosswalk_face: str = "#FFF0F5"
    crosswalk_edge: str = "#B88A9B"
    history: str = "#2563EB"
    ground_truth: str = "#0F766E"
    target_agent: str = "#2563EB"
    context_agent: str = "#CBD5E1"
    agent_edge: str = "#111827"
    prediction_cmap: str = "Oranges"


@dataclass(frozen=True)
class VisualizationConfig:
    root: str = ROOT
    processed: str = PROCESSED
    split: str = SPLIT
    map_root: str = MAP_ROOT
    checkpoint: str = CHECKPOINT
    output_dir: str = OUTPUT_DIR
    start_index: int = START_INDEX
    end_index: Optional[int] = END_INDEX
    categories: Tuple[int, ...] = CATEGORIES
    device: str = DEVICE
    batch_size: int = 1
    num_workers: int = 4
    num_historical_steps: int = 50
    num_future_steps: int = 60
    top_k: int = 6
    draw_context_history: bool = True
    draw_context_future: bool = False
    draw_predictions: bool = True
    dpi: int = 220
    figure_width: float = 24.0
    figure_height: float = 20.0
    strict_checkpoint: bool = False


class MoCARInference:
    def __init__(self, checkpoint: str, device: str, strict: bool = False) -> None:
        self.device = torch.device(device)
        self.model = Net()
        self._load_checkpoint(checkpoint=checkpoint, strict=strict)
        self.model.to(self.device).eval()

    def _load_checkpoint(self, checkpoint: str, strict: bool) -> None:
        result = load_weights(self.model, checkpoint, strict=strict)
        if result.missing_keys or result.unexpected_keys:
            print(
                "[checkpoint] loaded with "
                f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}"
            )

    @torch.no_grad()
    def predict(self, data: HeteroData) -> Tuple[HeteroData, dict]:
        data = data.to(self.device)
        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
        pred = self.model(data)
        return data.cpu(), {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in pred.items()}


class AV2MapRenderer:
    def __init__(self, map_root: str, style: VisualizationStyle) -> None:
        self.map_root = Path(map_root) if map_root else None
        self.style = style

    def draw(self, ax: plt.Axes, scenario_id: str) -> None:
        if self.map_root is None or ArgoverseStaticMap is None:
            return
        map_dir = self.map_root / str(scenario_id)
        if not map_dir.exists():
            return
        avm = ArgoverseStaticMap.from_map_dir(map_dir)
        for lane_segment in avm.get_scenario_lane_segments():
            ax.add_patch(
                Polygon(
                    lane_segment.polygon_boundary[:, :2],
                    closed=True,
                    facecolor=self.style.lane_face,
                    edgecolor=self.style.lane_edge,
                    linewidth=0.8,
                    alpha=0.82,
                    zorder=0,
                )
            )
        for crossing in avm.get_scenario_ped_crossings():
            ax.add_patch(
                Polygon(
                    crossing.polygon[:, :2],
                    closed=True,
                    facecolor=self.style.crosswalk_face,
                    edgecolor=self.style.crosswalk_edge,
                    linewidth=0.25,
                    alpha=0.85,
                    zorder=1,
                )
            )


class ScenarioRenderer:
    VEHICLE_TYPES = {0, 4, 5, 6, 7}
    PEDESTRIAN_TYPES = {1, 2, 3, 8}

    def __init__(self, config: VisualizationConfig, style: VisualizationStyle) -> None:
        self.config = config
        self.style = style
        base_cmap = cm.get_cmap(style.prediction_cmap)
        self.prediction_cmap = mcolors.LinearSegmentedColormap.from_list(
            "mocar_prediction_confidence",
            base_cmap(np.linspace(0.22, 1.0, 256)),
        )
        self.norm = mcolors.Normalize(vmin=0.0, vmax=0.7)
        self.map_renderer = AV2MapRenderer(config.map_root, style)

    def render(self, data: HeteroData, pred: dict, step: int, output_dir: Path) -> None:
        scenario_id = str(data["scenario_id"][0]) if isinstance(data["scenario_id"], Sequence) else str(data["scenario_id"])
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        fig.patch.set_facecolor(self.style.background)
        ax.set_facecolor(self.style.background)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")

        self.map_renderer.draw(ax, scenario_id)

        position = data["agent"]["position"][..., :2].numpy()
        heading = data["agent"]["heading"].numpy()
        valid_mask = data["agent"]["valid_mask"].numpy().astype(bool)
        agent_type = data["agent"]["type"].numpy()
        category = data["agent"]["category"].numpy()
        target_mask = np.isin(category, np.asarray(self.config.categories))

        self._draw_trajectories(ax, position, valid_mask, target_mask)
        if self.config.draw_predictions:
            self._draw_predictions(ax, pred, target_mask)
        self._draw_agents(ax, position, heading, valid_mask, agent_type, target_mask)
        self._draw_colorbar(fig, ax)

        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{step:06d}_{scenario_id}.png", dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

    def _draw_trajectories(
        self,
        ax: plt.Axes,
        position: np.ndarray,
        valid_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> None:
        H = self.config.num_historical_steps
        for idx in range(position.shape[0]):
            is_target = bool(target_mask[idx])
            if not is_target and not self.config.draw_context_history:
                continue
            hist_valid = valid_mask[idx, :H]
            if hist_valid.any():
                alpha = 0.88 if is_target else 0.26
                lw = 4.0 if is_target else 1.8
                ax.plot(
                    position[idx, :H][hist_valid, 0],
                    position[idx, :H][hist_valid, 1],
                    color=self.style.history,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=4 if is_target else 2,
                )

            if not is_target and not self.config.draw_context_future:
                continue
            fut_valid = valid_mask[idx, H:H + self.config.num_future_steps]
            if fut_valid.any():
                fut = position[idx, H:H + self.config.num_future_steps][fut_valid]
                ax.plot(
                    fut[:, 0],
                    fut[:, 1],
                    color=self.style.ground_truth,
                    linewidth=6.0 if is_target else 2.0,
                    alpha=0.92 if is_target else 0.22,
                    zorder=5 if is_target else 2,
                )

    def _draw_predictions(self, ax: plt.Axes, pred: dict, target_mask: np.ndarray) -> None:
        future_traj = pred["future_traj"].numpy()
        mode_prob = F.softmax(pred["mode_logits"], dim=-1).numpy()
        target_indices = np.flatnonzero(target_mask)
        for agent_idx in target_indices:
            probs = mode_prob[agent_idx]
            order = np.argsort(probs)[::-1][: self.config.top_k]
            for rank, mode_idx in enumerate(order[::-1]):
                prob = float(probs[mode_idx])
                traj = future_traj[agent_idx, mode_idx, :, :2]
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=self.prediction_cmap(self.norm(prob)),
                    linewidth=3.2 + 2.2 * prob,
                    alpha=0.58 + 0.34 * prob,
                    zorder=6 + rank,
                )

    def _draw_agents(
        self,
        ax: plt.Axes,
        position: np.ndarray,
        heading: np.ndarray,
        valid_mask: np.ndarray,
        agent_type: np.ndarray,
        target_mask: np.ndarray,
    ) -> None:
        H = self.config.num_historical_steps - 1
        for idx in range(position.shape[0]):
            if not valid_mask[idx, H]:
                continue
            center_x, center_y = position[idx, H]
            is_target = bool(target_mask[idx])
            facecolor = self.style.target_agent if is_target else self.style.context_agent
            alpha = 0.95 if is_target else 0.72
            zorder = 12 if is_target else 8
            if int(agent_type[idx]) in self.VEHICLE_TYPES:
                self._draw_vehicle(ax, center_x, center_y, heading[idx, H], facecolor, alpha, zorder)
            elif int(agent_type[idx]) in self.PEDESTRIAN_TYPES:
                self._draw_point_agent(ax, center_x, center_y, facecolor, alpha, zorder)
            else:
                self._draw_point_agent(ax, center_x, center_y, facecolor, alpha, zorder)

    def _draw_vehicle(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        heading: float,
        facecolor: str,
        alpha: float,
        zorder: int,
    ) -> None:
        width, height = 1.85, 4.35
        patch = Rectangle(
            (x - width / 2.0, y - height / 2.0),
            width,
            height,
            facecolor=facecolor,
            edgecolor=self.style.agent_edge,
            linewidth=1.1,
            alpha=alpha,
            zorder=zorder,
        )
        patch.set_transform(
            transforms.Affine2D().rotate_deg_around(x, y, np.degrees(heading) + 90.0) + ax.transData
        )
        ax.add_patch(patch)

    def _draw_point_agent(
        self,
        ax: plt.Axes,
        x: float,
        y: float,
        facecolor: str,
        alpha: float,
        zorder: int,
    ) -> None:
        ax.add_patch(
            Circle(
                (x, y),
                radius=0.62,
                facecolor=facecolor,
                edgecolor=self.style.agent_edge,
                linewidth=1.0,
                alpha=alpha,
                zorder=zorder,
            )
        )

    def _draw_colorbar(self, fig: plt.Figure, ax: plt.Axes) -> None:
        if not self.config.draw_predictions:
            return
        sm = ScalarMappable(cmap=self.prediction_cmap, norm=self.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.024, pad=0.006)
        cbar.set_label("Mode probability", fontsize=18, labelpad=8)
        cbar.ax.tick_params(labelsize=15, width=0.8, length=3)


def parse_categories(raw: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render MoCAR predictions on AV2 scenarios.")
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--processed", type=str, default=PROCESSED)
    parser.add_argument("--split", type=str, default=SPLIT)
    parser.add_argument("--map_root", type=str, default=MAP_ROOT)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--start_index", type=int, default=START_INDEX)
    parser.add_argument("--end_index", type=int, default=END_INDEX)
    parser.add_argument("--categories", type=parse_categories, default=CATEGORIES)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--strict_checkpoint", action="store_true")
    parser.add_argument("--hide_context_history", action="store_true")
    parser.add_argument("--draw_context_future", action="store_true")
    parser.add_argument("--hide_predictions", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> VisualizationConfig:
    return VisualizationConfig(
        root=args.root,
        processed=args.processed,
        split=args.split,
        map_root=args.map_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        categories=args.categories,
        device=args.device,
        num_workers=args.num_workers,
        top_k=args.top_k,
        dpi=args.dpi,
        strict_checkpoint=args.strict_checkpoint,
        draw_context_history=not args.hide_context_history,
        draw_context_future=args.draw_context_future,
        draw_predictions=not args.hide_predictions,
    )


def iter_selected_batches(loader: DataLoader, start: int, end: Optional[int]) -> Iterable[Tuple[int, Batch]]:
    for step, data in enumerate(tqdm(loader, smoothing=0.1)):
        if step < start:
            continue
        if end is not None and step >= end:
            break
        yield step, data


def visualize(config: VisualizationConfig) -> Path:
    dataset = AVDataset(
        root=config.root,
        processed=config.processed,
        split=config.split,
        transform=TargetBuilder(config.num_historical_steps, config.num_future_steps),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_dir = Path(config.output_dir) / f"{config.split}_{timestamp}"

    inference = MoCARInference(
        checkpoint=config.checkpoint,
        device=config.device,
        strict=config.strict_checkpoint,
    )
    renderer = ScenarioRenderer(config=config, style=VisualizationStyle())

    for step, data in iter_selected_batches(loader, config.start_index, config.end_index):
        data_cpu, pred = inference.predict(data)
        renderer.render(data_cpu, pred, step=step, output_dir=output_dir)

    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    output_dir = visualize(config_from_args(args))
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
