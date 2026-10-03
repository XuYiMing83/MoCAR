from argparse import ArgumentParser
from inspect import signature

import pytorch_lightning as pl
import torch

from datamodules import ArgoverseV2DataModule
from model_MoCAR.checkpoint import load_weights
from model_MoCAR.Net import Net


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root containing the processed directory.")
    parser.add_argument("--processed", type=str, default="MoCAR_data", help="Processed data directory name.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strict_checkpoint", action="store_true")
    Net.add_model_specific_args(parser)
    return parser


def main():
    torch.set_float32_matmul_precision("high")
    args = build_parser().parse_args()

    datamodule = ArgoverseV2DataModule(
        root=args.root,
        processed=args.processed,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_keys = set(signature(Net.__init__).parameters)
    model_keys.discard("self")
    model_kwargs = {key: value for key, value in vars(args).items() if key in model_keys}
    model = Net(**model_kwargs)
    load_result = load_weights(model, args.ckpt_path, strict=args.strict_checkpoint)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "[checkpoint] loaded with "
            f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
        )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.validate(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
