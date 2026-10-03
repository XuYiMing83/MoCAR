import os
import pickle
import warnings
import zlib
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from Datasets.dataset_pkl import AVDataset


warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")


def build_parser():
    parser = ArgumentParser(
        description="Generate compressed trajectory segments for chunk VAE pretraining."
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--processed", type=str, default="MoCAR_data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Set future valid_mask entries to False before computing segment ranges.",
    )
    return parser


def build_chunk_item(data, history_only=False):
    valid_mask = data["agent"]["valid_mask"].clone()
    if history_only:
        valid_mask[:, 50:] = False

    has_true = valid_mask.any(dim=1)
    first_idx = valid_mask.float().argmax(dim=1)
    last_idx = valid_mask.size(1) - 1 - valid_mask.flip(dims=[1]).float().argmax(dim=1)
    true_idx = torch.stack([first_idx, last_idx], dim=1)

    length = valid_mask.sum(dim=1)
    contiguous_length = torch.where(
        has_true,
        last_idx - first_idx + 1,
        torch.zeros_like(length),
    )
    hole_mask = length != contiguous_length
    if hole_mask.any():
        raise RuntimeError(f"{data['scenario_id']} has non-contiguous valid_mask segments.")

    return {
        "scenario_id": data["scenario_id"],
        "agent": {
            "valid_mask": valid_mask.detach().cpu(),
            "position": data["agent"]["position"][:, :, :2].detach().cpu(),
            "heading": data["agent"]["heading"].detach().cpu(),
            "velocity": data["agent"]["velocity"][:, :, :2].detach().cpu(),
            "type": data["agent"]["type"].detach().cpu(),
            "true_idx": true_idx.detach().cpu(),
            "length": length.detach().cpu(),
        },
    }


def main():
    args = build_parser().parse_args()
    dataset = AVDataset(root=args.root, processed=args.processed, split=args.split)
    out = []
    for data in tqdm(dataset, smoothing=0.1):
        item = build_chunk_item(data, history_only=args.history_only)
        out.append(zlib.compress(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(out)} chunk VAE pretraining samples to {args.output}")


if __name__ == "__main__":
    main()
