from argparse import ArgumentParser
from inspect import signature

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from model_MoCAR.Net import Net


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root containing the processed directory.")
    parser.add_argument("--processed", type=str, default="MoCAR_data", help="Processed data directory name.")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    Net.add_model_specific_args(parser)
    return parser


def main():
    torch.set_float32_matmul_precision("high")
    args = build_parser().parse_args()
    pl.seed_everything(2026, workers=True)

    datamodule = ArgoverseV2DataModule(
        root=args.root,
        processed=args.processed,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )

    model_keys = set(signature(Net.__init__).parameters)
    model_keys.discard("self")
    model_kwargs = {key: value for key, value in vars(args).items() if key in model_keys}
    model = Net(**model_kwargs)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            monitor="val_minFDE",
            mode="min",
            save_top_k=5,
            save_last=True,
            filename="{epoch:03d}-{val_minFDE:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else "auto",
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
