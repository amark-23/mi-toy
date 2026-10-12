"""Permutation-invariant circuit fingerprints for trained models."""

import torch

from analysis import offset_profile
from attribution import cancellation, cancellation_slope, logit_attribution, summarize
from data import make_batch
from hooks import run_with_cache
from model import Config, Transformer


def layer1_peaks(
    model: Transformer, cfg: Config, half_len: int, n_seq: int = 256
) -> list[tuple[int, int, float]]:
    """Peak attention offset and its weight for each layer-1 head (second-half queries).

    Returns:
        List of (head, peak_offset, peak_weight), one entry per layer-1 head.
    """
    tokens = make_batch(n_seq, half_len, cfg.vocab_size)
    _, cache = run_with_cache(model, tokens)
    pattern = cache["layers.1.hook_pattern"]
    profile = offset_profile(pattern, half_len, 2 * half_len - 1, (3 * half_len - 1) // 2)
    weights, offsets = profile.max(dim=-1)
    return [(h, offsets[h].item(), weights[h].item()) for h in range(cfg.n_heads)]


def logit_fingerprint(model: Transformer, cfg: Config, half_len: int = 20, n_seq: int = 256) -> dict:
    """Permutation-invariant logit summary of one model.

    Returns:
        Dict with 'heads' (layer-1 heads as (second-half DLA, first-half RMS),
        sorted by DLA, strongest first), 'total' (second-half centered logit),
        'layer0_rms' (first-half RMS of layer 0, heads added in quadrature) and
        'beta' (slope of the layer-1 sum on the layer-0 sum, first half).
    """
    tokens = make_batch(n_seq, half_len, cfg.vocab_size)
    contrib = logit_attribution(model, tokens)
    rows = summarize(contrib, half_len)
    layer0 = [f"L0H{h}" for h in range(cfg.n_heads)]
    layer1 = [f"L1H{h}" for h in range(cfg.n_heads)]
    heads = sorted(((rows[n][1], rows[n][2]) for n in layer1), reverse=True)
    total = sum(r[1] for r in rows.values())
    _, layer0_rms = cancellation(contrib, layer0, half_len)
    _, beta = cancellation_slope(contrib, layer0, layer1, half_len)
    return {"heads": heads, "total": total, "layer0_rms": layer0_rms, "beta": beta}


def head_roles(
    model: Transformer, cfg: Config, half_len: int = 20, n_seq: int = 256
) -> list[tuple[float, float, int, float]]:
    """Attention peak and logit contribution of the same layer-1 head, side by side.

    Returns:
        List of (second-half DLA, first-half RMS, peak_offset, peak_weight), one per
        layer-1 head, sorted by DLA, strongest first.
    """
    peaks = layer1_peaks(model, cfg, half_len, n_seq)
    tokens = make_batch(n_seq, half_len, cfg.vocab_size)
    rows = summarize(logit_attribution(model, tokens), half_len)
    roles = []
    for head, offset, weight in peaks:
        _, dla, rms = rows[f"L1H{head}"]
        roles.append((dla, rms, offset, weight))
    return sorted(roles, reverse=True)


if __name__ == "__main__":
    from evaluate import load_model
    from utils import set_seed

    cfg = Config()
    plans = {"varlen": (20, 10), "fixed": (20, 15)}
    for kind, lengths in plans.items():
        print(f"\n{kind}: layer-1 heads as peak_offset(weight) at length {lengths[0]} > length {lengths[1]}")
        for seed in range(5):
            model = load_model(f"part1_toy/checkpoints/{kind}_seed{seed}.pt", cfg)
            per_length = []
            for half_len in lengths:
                set_seed(0)
                per_length.append(layer1_peaks(model, cfg, half_len))
            rows = sorted(zip(*per_length), key=lambda r: -r[0][2])
            line = "  ".join(
                f"{a[1]:2d}({a[2]:.2f})>{b[1]:2d}({b[2]:.2f})" for a, b in rows
            )
            print(f"  seed {seed} | {line}")

    for kind in plans:
        print(f"\n{kind}: total | layer-0 RMS | beta | layer-1 heads as DLA/RMS, strongest first")
        for seed in range(5):
            model = load_model(f"part1_toy/checkpoints/{kind}_seed{seed}.pt", cfg)
            set_seed(0)
            fp = logit_fingerprint(model, cfg)
            heads = "  ".join(f"{d:6.2f}/{r:5.2f}" for d, r in fp["heads"])
            print(
                f"  seed {seed} | {fp['total']:6.2f} | {fp['layer0_rms']:6.2f} | "
                f"{fp['beta']:6.3f} | {heads}"
            )

    print("\nroles: layer-1 heads as DLA/RMS@offset(weight), strongest DLA first")
    for kind in plans:
        print(f"{kind}:")
        for seed in range(5):
            model = load_model(f"part1_toy/checkpoints/{kind}_seed{seed}.pt", cfg)
            set_seed(0)
            cells = [f"{d:6.2f}/{r:5.2f}@{o:2d}({w:.2f})" for d, r, o, w in head_roles(model, cfg)]
            print(f"  seed {seed} | " + "  ".join(cells))
