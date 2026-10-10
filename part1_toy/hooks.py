"""Helpers to read activations through the model's hook points."""

import torch

from model import Transformer


def run_with_cache(
    model: Transformer, tokens: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the model and save every attention pattern and per-head result.

    Args:
        model: The Transformer to run.
        tokens: Integer tensor of shape (batch, pos).

    Returns:
        The logits and a cache: a dict from hook name (for example
        "layers.0.hook_pattern") to the tensor that passed through it.
    """
    cache = {}
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            cache[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if name.endswith("hook_pattern") or name.endswith("hook_z"):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            logits = model(tokens)
    finally:
        for handle in handles:
            handle.remove()

    return logits, cache


if __name__ == "__main__":
    from data import make_batch
    from evaluate import load_model
    from model import Config

    cfg = Config()
    model = load_model("part1_toy/varlen_seed0.pt", cfg)
    tokens = make_batch(4, 10, cfg.vocab_size)
    logits, cache = run_with_cache(model, tokens)
    for name, tensor in cache.items():
        print(name, tuple(tensor.shape))