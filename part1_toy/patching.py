import torch

from data import make_batch
from hooks import run_with_cache

def make_corrupted(tokens: torch.Tensor, half_len: int, vocab_size: int) -> torch.Tensor:
    """Replace the first half with a fresh random block, keeping the second half.

    Args:
        tokens: Clean sequences of shape (batch, 2 * half_len).
        half_len: Repeat length.
        vocab_size: Number of distinct tokens.

    Returns:
        Tensor of the same shape: [new random block | original second half].
    """
    corrupted = tokens.clone()
    fresh = make_batch(tokens.shape[0], half_len, vocab_size)
    corrupted[:, :half_len] = fresh[:, :half_len]
    return corrupted


def correct_logit(logits: torch.Tensor, tokens: torch.Tensor, half_len: int) -> float:
    """Mean centered logit of the true next token over the second-half positions."""
    centered = logits - logits.mean(dim=-1, keepdim=True)
    targets = tokens[:, 1:].unsqueeze(-1)
    picked = centered[:, :-1].gather(-1, targets).squeeze(-1)
    return picked[:, half_len:].mean().item()

def make_patch_hook(head: int, clean_z: torch.Tensor, positions: slice):
    """Hook that overwrites one head's output at some positions with clean values."""

    def hook(module, inputs, output):
        output = output.clone()
        output[:, head, positions] = clean_z[:, head, positions]
        return output

    return hook


def run_with_patch(
    model,
    tokens: torch.Tensor,
    clean_cache: dict[str, torch.Tensor],
    patches: list[tuple[int, int, slice]],
) -> torch.Tensor:
    """Run the model on `tokens`, patching (layer, head, positions) from the clean cache."""
    handles = []
    for layer, head, positions in patches:
        module = model.layers[layer].hook_z
        clean_z = clean_cache[f"layers.{layer}.hook_z"]
        handles.append(
            module.register_forward_hook(make_patch_hook(head, clean_z, positions))
        )
    try:
        with torch.no_grad():
            logits = model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    return logits






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
        set_seed(0)
        clean = make_batch(256, half_len, cfg.vocab_size)
        set_seed(1)
        corrupted = make_corrupted(clean, half_len, cfg.vocab_size)
        with torch.no_grad():
            m_clean = correct_logit(model(clean), clean, half_len)
            m_corrupt = correct_logit(model(corrupted), corrupted, half_len)
        print(f"{name}: clean {m_clean:8.3f} | corrupted {m_corrupt:8.3f}")
        _, clean_cache = run_with_cache(model, clean)
        everything = [
            (layer, head, slice(None))
            for layer in range(cfg.n_layers)
            for head in range(cfg.n_heads)
        ]
        patched = run_with_patch(model, corrupted, clean_cache, everything)
        m_all = correct_logit(patched, corrupted, half_len)
        print(f"   patch all heads, all positions: {m_all:8.3f} (should equal clean)")
        ranges = {
            "first half": slice(0, half_len),
            "second half": slice(half_len, None),
            "all": slice(None),
        }
        print("   head   | " + " | ".join(f"{k:>11s}" for k in ranges))
        for layer in range(cfg.n_layers):
            for head in range(cfg.n_heads):
                cells = []
                for positions in ranges.values():
                    patched = run_with_patch(
                        model, corrupted, clean_cache, [(layer, head, positions)]
                    )
                    m = correct_logit(patched, corrupted, half_len)
                    cells.append((m - m_corrupt) / (m_clean - m_corrupt))
                print(f"   L{layer}H{head}   | " + " | ".join(f"{c:11.3f}" for c in cells))

        _, corrupt_cache = run_with_cache(model, corrupted)
        print("   noising: drop when the corrupted z is put into the clean run")
        print("   head   | " + " | ".join(f"{k:>11s}" for k in ranges))
        for layer in range(cfg.n_layers):
            for head in range(cfg.n_heads):
                cells = []
                for positions in ranges.values():
                    noised = run_with_patch(
                        model, clean, corrupt_cache, [(layer, head, positions)]
                    )
                    m = correct_logit(noised, clean, half_len)
                    cells.append((m_clean - m) / (m_clean - m_corrupt))
                print(f"   L{layer}H{head}   | " + " | ".join(f"{c:11.3f}" for c in cells))
                layer0 = [(0, h) for h in range(cfg.n_heads)]
        for label, positions in ranges.items():
            patches = [(layer, head, positions) for layer, head in layer0]
            denoised = run_with_patch(model, corrupted, clean_cache, patches)
            noised = run_with_patch(model, clean, corrupt_cache, patches)
            rec = (correct_logit(denoised, corrupted, half_len) - m_corrupt) / (m_clean - m_corrupt)
            drop = (m_clean - correct_logit(noised, clean, half_len)) / (m_clean - m_corrupt)
            print(f"   all L0 heads, {label:11s}: recovery {rec:6.3f} | drop {drop:6.3f}")      