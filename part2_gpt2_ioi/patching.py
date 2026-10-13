"""Activation patching of attention-head outputs on the IOI task (last position)."""

import torch

from ioi_data import NAMES, build_batch, logit_diff, make_prompts


def make_patch_hook(heads, source_z, pos=-1):
    """Hook that overwrites `heads` of hook_z at position `pos` with values from source_z."""

    def fn(z, hook):
        z = z.clone()
        z[:, pos, heads] = source_z[:, pos, heads]
        return z

    return fn


def run_patched(model, tokens, source_cache, patches):
    """Run `tokens` with head outputs patched from `source_cache`.

    Args:
        patches: list of (layer, [heads]).

    Returns:
        Logits of the patched run.
    """
    hooks = [
        (f"blocks.{layer}.attn.hook_z", make_patch_hook(heads, source_cache[f"blocks.{layer}.attn.hook_z"]))
        for layer, heads in patches
    ]
    with torch.no_grad():
        return model.run_with_hooks(tokens, fwd_hooks=hooks)


def head_sweep(model, tokens, source_cache, io, s, n_layers=12, n_heads=12):
    """LD of the run on `tokens` with each single head patched from `source_cache`.

    Returns:
        Tensor (n_layers, n_heads, batch) of logit differences.
    """
    out = torch.zeros(n_layers, n_heads, tokens.shape[0])
    for layer in range(n_layers):
        for head in range(n_heads):
            logits = run_patched(model, tokens, source_cache, [(layer, [head])])
            out[layer, head] = logit_diff(logits, io, s)
        print(f"  layer {layer} done", flush=True)
    return out


def show_ranking(title, values, k_top=10, k_bottom=5):
    flat = [(values[l, h].item(), f"L{l}H{h}") for l in range(values.shape[0]) for h in range(values.shape[1])]
    flat.sort(reverse=True)
    print(title)
    print("  top:   " + "  ".join(f"{n} {v:.3f}" for v, n in flat[:k_top]))
    print("  bottom:" + "  ".join(f"{n} {v:.3f}" for v, n in flat[::-1][:k_bottom]))


if __name__ == "__main__":
    from transformer_lens.model_bridge import TransformerBridge

    model = TransformerBridge.boot_transformers("gpt2")
    model.enable_compatibility_mode()
    batch = build_batch(model, make_prompts(900, NAMES, seed=0), 0)
    io, s = batch["io"], batch["s"]
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(batch["tokens"])
        corrupt_logits, corrupt_cache = model.run_with_cache(batch["corrupt"])
    clean_ld = logit_diff(clean_logits, io, s)
    corrupt_ld = logit_diff(corrupt_logits, io, s)
    gap = clean_ld - corrupt_ld
    print(f"mean LD clean {clean_ld.mean():.3f}, corrupted {corrupt_ld.mean():.3f}")
    print(f"gap per prompt: mean {gap.mean():.3f}, min {gap.min():.3f}, max {gap.max():.3f}")

    every_head = [(layer, list(range(12))) for layer in range(12)]
    same = run_patched(model, batch["corrupt"], corrupt_cache, every_head)
    print("identity (corrupt patched with itself), max logit change:", (same - corrupt_logits)[:, -1].abs().max().item())
    full = run_patched(model, batch["corrupt"], clean_cache, every_head)
    print("all heads patched with clean, max logit change vs clean:", (full - clean_logits)[:, -1].abs().max().item())
    print("  LD of that run:", logit_diff(full, io, s).mean().item())

    import os

    import numpy as np

    print("denoising sweep (corrupted run, clean head output at the last position):")
    denoise = head_sweep(model, batch["corrupt"], clean_cache, io, s)
    recovery = (denoise.mean(-1) - corrupt_ld.mean()) / gap.mean()
    print("noising sweep (clean run, corrupted head output at the last position):")
    noise = head_sweep(model, batch["tokens"], corrupt_cache, io, s)
    drop = (clean_ld.mean() - noise.mean(-1)) / gap.mean()
    show_ranking("recovery (denoising), fraction of the gap:", recovery)
    show_ranking("drop (noising), fraction of the gap:", drop)
    os.makedirs("part2_gpt2_ioi/results", exist_ok=True)
    np.savetxt("part2_gpt2_ioi/results/patch_last_recovery.csv", recovery.numpy(), delimiter=",", fmt="%.5f")
    np.savetxt("part2_gpt2_ioi/results/patch_last_drop.csv", drop.numpy(), delimiter=",", fmt="%.5f")
    print("saved part2_gpt2_ioi/results/patch_last_recovery.csv and patch_last_drop.csv")
