"""Plots for inspecting the trained models."""

import matplotlib.pyplot as plt

from model import Config


def plot_patterns(cache: dict, cfg: Config, half_len: int, path: str) -> None:
    """Plot every head's attention pattern for the first sequence in the cache.

    Args:
        cache: Output of run_with_cache.
        cfg: Model configuration.
        half_len: Repeat length of the sequence (draws the boundary lines).
        path: Where to save the image.
    """
    fig, axes = plt.subplots(
        cfg.n_layers, cfg.n_heads, figsize=(3 * cfg.n_heads, 3 * cfg.n_layers)
    )
    for layer in range(cfg.n_layers):
        pattern = cache[f"layers.{layer}.hook_pattern"][0]  # (head, query_pos, key_pos)
        for head in range(cfg.n_heads):
            ax = axes[layer, head]
            ax.imshow(pattern[head].numpy(), cmap="viridis", vmin=0, vmax=1)
            ax.axhline(half_len - 0.5, color="white", linewidth=0.5)
            ax.axvline(half_len - 0.5, color="white", linewidth=0.5)
            ax.set_title(f"L{layer}H{head}")
    fig.supxlabel("key position (attended to)")
    fig.supylabel("query position (attending)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from data import make_batch
    from evaluate import load_model
    from hooks import run_with_cache
    from utils import set_seed

    cfg = Config()
    runs = [
        ("fixed", "part1_toy/seed0.pt", 20),
        ("varlen", "part1_toy/varlen_seed0.pt", 20),
        ("varlen", "part1_toy/varlen_seed0.pt", 10),
    ]
    for name, checkpoint, half_len in runs:
        set_seed(0)
        tokens = make_batch(1, half_len, cfg.vocab_size)
        model = load_model(checkpoint, cfg)
        _, cache = run_with_cache(model, tokens)
        plot_patterns(cache, cfg, half_len, f"figures/patterns_{name}_len{half_len}.png")