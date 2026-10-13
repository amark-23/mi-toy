"""Direct logit attribution of the IOI logit difference in GPT-2 small."""

import torch

from ioi_data import NAMES, build_batch, logit_diff, make_prompts


def ld_components(model, cache, io, s, n_layers=12, n_heads=12):
    """Exact split of logit(IO) - logit(S) at the last position, per component.

    Returns:
        Dict name -> (batch,) tensor. Names: embed, pos_embed, L{l}H{h}, L{l}b_O,
        L{l}MLP and b_U (constant). The values add up to the logit difference.
    """
    d = (model.W_U[:, io] - model.W_U[:, s]).T  # (batch, d_model)
    scale = cache["ln_final.hook_scale"][:, -1, :]  # (batch, 1)

    def contrib(vec):
        vec = vec - vec.mean(-1, keepdim=True)
        return (vec / scale * d).sum(-1)

    batch = io.shape[0]
    out = {
        "embed": contrib(cache["hook_embed"][:, -1]),
        "pos_embed": contrib(cache["hook_pos_embed"][:, -1]),
    }
    for layer in range(n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"][:, -1]  # (batch, head, d_head)
        head_out = torch.einsum("bhd,hdm->bhm", z, model.W_O[layer])
        for h in range(n_heads):
            out[f"L{layer}H{h}"] = contrib(head_out[:, h])
        out[f"L{layer}b_O"] = contrib(model.b_O[layer].expand(batch, -1))
        out[f"L{layer}MLP"] = contrib(cache[f"blocks.{layer}.hook_mlp_out"][:, -1])
    out["b_U"] = model.b_U[io] - model.b_U[s]
    return out


if __name__ == "__main__":
    from transformer_lens.model_bridge import TransformerBridge

    model = TransformerBridge.boot_transformers("gpt2")
    model.enable_compatibility_mode()
    batch = build_batch(model, make_prompts(900, NAMES, seed=0), 0)
    with torch.no_grad():
        logits, cache = model.run_with_cache(batch["tokens"])
    comps = ld_components(model, cache, batch["io"], batch["s"])
    total = sum(comps.values())
    ld = logit_diff(logits, batch["io"], batch["s"])
    print("prompts:", len(ld), "| mean LD:", ld.mean().item())
    print("max |sum of components - LD|:", (total - ld).abs().max().item())

    names = list(comps)
    mat = torch.stack([comps[n] for n in names])  # (component, prompt)
    mean, std = mat.mean(1), mat.std(1)
    heads = [i for i, n in enumerate(names) if n[0] == "L" and "H" in n]
    order = sorted(heads, key=lambda i: mean[i].item(), reverse=True)
    print("\ntop positive heads (mean contribution to LD, std over prompts):")
    for i in order[:10]:
        print(f"  {names[i]:7s} {mean[i]:6.3f} +- {std[i]:.3f}")
    print("top negative heads:")
    for i in order[::-1][:8]:
        print(f"  {names[i]:7s} {mean[i]:6.3f} +- {std[i]:.3f}")
    groups = {
        "embed + pos_embed": ["embed", "pos_embed"],
        "b_U": ["b_U"],
        "attention biases": [n for n in names if n.endswith("b_O")],
        "MLPs (all layers)": [n for n in names if n.endswith("MLP")],
        "all heads": [names[i] for i in heads],
    }
    print("\ngroup totals:")
    for label, members in groups.items():
        print(f"  {label:18s} {sum(comps[m] for m in members).mean():7.3f}")
    print("  sum of top 3 positive heads:", sum(mean[i].item() for i in order[:3]), "| total LD:", ld.mean().item())
    print("  largest MLPs:", sorted(((comps[n].mean().item(), n) for n in names if n.endswith("MLP")), reverse=True)[:3])
