"""Numerical analysis of attention patterns."""

import torch


def offset_profile(
    pattern: torch.Tensor, q_start: int, q_end: int, max_offset: int
) -> torch.Tensor:
    """Mean attention weight at each distance back, per head.

    Args:
        pattern: Attention patterns, shape (batch, head, query_pos, key_pos).
        q_start: First query position to include.
        q_end: One past the last query position to include.
        max_offset: Largest distance back to measure.

    Returns:
        Tensor of shape (head, max_offset + 1); entry [h, d] is the mean weight
        head h puts on the key d positions before the query.
    """
    profile = torch.zeros(pattern.shape[1], max_offset + 1)
    for d in range(max_offset + 1):
        # Weight on the key d positions back, for every query in the range
        weights = [pattern[:, :, q, q - d] for q in range(q_start, q_end) if q - d >= 0]
        profile[:, d] = torch.stack(weights, dim=-1).mean(dim=(0, -1))
    return profile


if __name__ == "__main__":
    from data import make_batch
    from evaluate import load_model
    from hooks import run_with_cache
    from model import Config
    from utils import set_seed

    cfg = Config()
    runs = [
        ("fixed", "part1_toy/seed0.pt", 20),
        ("fixed", "part1_toy/seed0.pt", 15),
        ("varlen", "part1_toy/varlen_seed0.pt", 20),
        ("varlen", "part1_toy/varlen_seed0.pt", 10),
    ]
    for name, checkpoint, half_len in runs:
        set_seed(0)
        tokens = make_batch(256, half_len, cfg.vocab_size)
        model = load_model(checkpoint, cfg)
        _, cache = run_with_cache(model, tokens)

        print(f"\n{name} model, repeat length {half_len}")
        for layer in range(cfg.n_layers):
            pattern = cache[f"layers.{layer}.hook_pattern"]
            profile = offset_profile(pattern, half_len, 2 * half_len - 1, 2 * half_len - 2)
            for head in range(cfg.n_heads):
                row = profile[head]
                print(
                    f"  L{layer}H{head} | self {row[0]:.2f} | prev {row[1]:.2f} "
                    f"| induction {row[half_len - 1]:.2f} | peak offset {row.argmax().item()}"
                    f"| peak offset {row.argmax().item()} ({row.max().item():.2f})"
                )