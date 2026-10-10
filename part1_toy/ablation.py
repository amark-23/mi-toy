"""Ablation experiments: how much does the loss depend on each head?"""

import torch

from data import make_batch
from hooks import run_with_ablation
from model import Transformer
from train import per_position_loss


def ablated_loss(
    model: Transformer,
    heads: list[tuple[int, int]],
    half_len: int,
    vocab_size: int,
    n_batches: int = 10,
    batch_size: int = 256,
) -> tuple[float, float]:
    """Mean first-half and second-half loss with the given heads switched off.

    Args:
        model: Trained model.
        heads: List of (layer, head) pairs to ablate; empty means no ablation.
        half_len: Repeat length of the test sequences.
        vocab_size: Number of distinct tokens.
        n_batches: Number of batches to average over.
        batch_size: Sequences per batch.

    Returns:
        Tuple (first_half_loss, second_half_loss).
    """
    first, second = [], []
    for _ in range(n_batches):
        tokens = make_batch(batch_size, half_len, vocab_size)
        logits = run_with_ablation(model, tokens, heads)
        losses = per_position_loss(logits, tokens)
        first.append(losses[:, : half_len - 1].mean().item())
        second.append(losses[:, half_len:].mean().item())
    return sum(first) / n_batches, sum(second) / n_batches

if __name__ == "__main__":
    from evaluate import load_model
    from model import Config
    from utils import set_seed

    cfg = Config()
    half_len = 20
    models = {
        "fixed": "part1_toy/seed0.pt",
        "varlen": "part1_toy/varlen_seed0.pt",
    }

    for name, checkpoint in models.items():
        model = load_model(checkpoint, cfg)
        print(f"\n{name} model, repeat length {half_len}")

        experiments = [("baseline", [])]
        experiments += [
            (f"L{layer}H{head}", [(layer, head)])
            for layer in range(cfg.n_layers)
            for head in range(cfg.n_heads)
        ]
        for label, heads in experiments:
            set_seed(0)
            first, second = ablated_loss(model, heads, half_len, cfg.vocab_size)
            print(f"  ablate {label:8s} | first half {first:.3f} | second half {second:.3f}")