import torch

from hooks import run_with_cache
from model import Transformer


def decompose(model: Transformer, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the final residual stream into the contribution of each component.

    Returns:
        Dict from component name ('embed', 'pos_embed', 'L0H0', ...) to a tensor
        of shape (batch, pos, d_model). The tensors sum to the final residual.
    """
    _, cache = run_with_cache(model, tokens)
    with torch.no_grad():
        parts = {
            "embed": model.embed(tokens),
            "pos_embed": model.pos_embed(tokens).expand(tokens.shape[0], -1, -1),
        }
        for layer in range(model.cfg.n_layers):
            z = cache[f"layers.{layer}.hook_z"]
            out = torch.einsum("bhpe,hed->bhpd", z, model.layers[layer].W_O)
            for head in range(model.cfg.n_heads):
                parts[f"L{layer}H{head}"] = out[:, head]
    return parts

def logit_attribution(model: Transformer, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Contribution of each component to the correct next token's centered logit.

    Returns:
        Dict from component name to a tensor of shape (batch, pos - 1): entry
        [b, p] is how much the component pushed up the logit of tokens[b, p + 1]
        (relative to the average logit) at position p.
    """
    parts = decompose(model, tokens)
    with torch.no_grad():
        W_U = model.unembed.W_U
        W_U_centered = W_U - W_U.mean(dim=1, keepdim=True)
        targets = tokens[:, 1:]
        wanted = W_U_centered[:, targets]
        return {
            name: torch.einsum("bpd,dbp->bp", x[:, :-1], wanted)
            for name, x in parts.items()
        }

def cancellation(
    contrib: dict[str, torch.Tensor], names: list[str], half_len: int
) -> tuple[float, float]:
    """First-half RMS of the summed contributions vs. the RMS if they were uncorrelated.

    Returns:
        Tuple (rms_sum, rms_independent).
    """
    first = torch.stack([contrib[n][:, : half_len - 1] for n in names])
    rms_sum = first.sum(dim=0).pow(2).mean().sqrt().item()
    rms_independent = first.pow(2).mean(dim=(1, 2)).sum().sqrt().item()
    return rms_sum, rms_independent

def cancellation_slope(
    contrib: dict[str, torch.Tensor],
    names_a: list[str],
    names_b: list[str],
    half_len: int,
) -> tuple[float, float]:
    """Correlation r and regression slope beta of set B on set A (first half).

    Each set's contributions are summed over its components, then flattened to
    one number per (sequence, position).

    Returns:
        Tuple (r, beta); beta = Cov(a, b) / Var(a).
    """
    a = sum(contrib[n][:, : half_len - 1] for n in names_a).flatten()
    b = sum(contrib[n][:, : half_len - 1] for n in names_b).flatten()
    cov = torch.cov(torch.stack([a, b]))
    r = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    beta = (cov[0, 1] / cov[0, 0]).item()
    return r, beta

def summarize(
    contrib: dict[str, torch.Tensor], half_len: int
) -> dict[str, tuple[float, float, float]]:
    """Per component: (first-half mean, second-half mean, first-half RMS)."""
    rows = {}
    for name, c in contrib.items():
        first = c[:, : half_len - 1]
        second = c[:, half_len:]
        rows[name] = (
            first.mean().item(),
            second.mean().item(),
            first.pow(2).mean().sqrt().item(),
        )
    return rows


if __name__ == "__main__":
    from data import make_batch
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
        set_seed(0)
        tokens = make_batch(256, half_len, cfg.vocab_size)
        contrib = logit_attribution(model, tokens)
        rows = summarize(contrib, half_len)

        print(f"\n{name} model, repeat length {half_len}")
        print(f"  {'component':10s} | 1st-half mean | 2nd-half mean | 1st-half RMS")
        for comp, (f_mean, s_mean, f_rms) in rows.items():
            print(f"  {comp:10s} | {f_mean:13.3f} | {s_mean:13.3f} | {f_rms:12.3f}")
        total_second = sum(r[1] for r in rows.values())
        print(f"  {'total':10s} | {'':13s} | {total_second:13.3f} |")
        groups = {
            "layer 0": ["L0H0", "L0H1", "L0H2", "L0H3"],
            "layer 1": ["L1H0", "L1H1", "L1H2", "L1H3"],
            "all": list(contrib),
        }
        for label, names in groups.items():
            rms_sum, rms_indep = cancellation(contrib, names, half_len)
            print(
                f"  first-half RMS, {label:7s}: sum {rms_sum:7.3f} | "
                f"uncorrelated {rms_indep:7.3f} | ratio {rms_sum / rms_indep:5.2f}"
            )
        r, beta = cancellation_slope(contrib, groups["layer 0"], groups["layer 1"], half_len)
        print(f"  layer 1 vs layer 0: r = {r:6.3f} | beta = {beta:6.3f}")
        for head in groups["layer 1"]:
            r_h, beta_h = cancellation_slope(contrib, groups["layer 0"], [head], half_len)
            print(f"    {head}: r = {r_h:6.3f} | beta = {beta_h:6.3f}")