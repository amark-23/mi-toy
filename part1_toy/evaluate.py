"""Evaluation of the trained induction model."""

import torch

from data import make_batch
from model import Config, Transformer
from train import per_position_loss
from utils import set_seed


def load_model(path: str, cfg: Config) -> Transformer:
    """Load trained weights into a fresh Transformer.

    Args:
        path: Path to a saved state_dict (.pt file).
        cfg: The configuration the model was trained with.

    Returns:
        The model in evaluation mode.
    """
    model = Transformer(cfg)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def eval_loss(
    model: Transformer,
    half_len: int,
    vocab_size: int,
    n_batches: int = 10,
    batch_size: int = 256,
) -> tuple[float, float]:
    """Mean first-half and second-half loss for a given repeat length.

    Args:
        model: Trained model.
        half_len: Length of the random half (the repeat length).
        vocab_size: Number of distinct tokens.
        n_batches: Number of batches to average over.
        batch_size: Sequences per batch.

    Returns:
        Tuple (first_half_loss, second_half_loss).
    """
    first, second = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            tokens = make_batch(batch_size, half_len, vocab_size)
            losses = per_position_loss(model(tokens), tokens)
            first.append(losses[:, : half_len - 1].mean().item())
            second.append(losses[:, half_len:].mean().item())
    return sum(first) / n_batches, sum(second) / n_batches



if __name__ == "__main__":
    cfg = Config()
    checkpoints = {
        "fixed length": "part1_toy/seed0.pt",
        "variable length": "part1_toy/varlen_seed0.pt",
    }

    for name, path in checkpoints.items():
        set_seed(0)
        model = load_model(path, cfg)
        print(name)
        for half_len in [5, 10, 15, 20]:
            first, second = eval_loss(model, half_len, cfg.vocab_size)
            print(f"  half_len {half_len:2d} | first half {first:.3f} | second half {second:.3f}")