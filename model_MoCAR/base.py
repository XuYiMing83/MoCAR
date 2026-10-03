from abc import ABC

import torch.nn as nn

from model_MoCAR.graph import TokenGraphBuilder
from model_MoCAR.losses import ChunkReconstructionLoss
from model_MoCAR.tokenization import EndAlignedTrajectoryTokenizer


class TrajectoryTokenDecoderBase(
    EndAlignedTrajectoryTokenizer,
    TokenGraphBuilder,
    ChunkReconstructionLoss,
    nn.Module,
    ABC,
):
    """Abstract substrate for decoder-only trajectory-token rollouts."""

    pass


class DecoderOnlyTrajectoryModel(TrajectoryTokenDecoderBase):
    """Autoregressive trajectory model that consumes and emits latent motion tokens."""

    pass
