from typing import Mapping

import torch


def load_weights(model: torch.nn.Module, checkpoint_path: str, strict: bool = False):
    """Load either a weights-only file or a full Lightning checkpoint into a module."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state[key[len("model."):]] = value
        else:
            cleaned_state[key] = value
    return model.load_state_dict(cleaned_state, strict=strict)
