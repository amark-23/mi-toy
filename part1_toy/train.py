"""Training utilities for the induction task."""
import random
import torch
import torch.nn.functional as F

from data import make_batch
from model import Config, Transformer
from utils import set_seed


def per_position_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of each next-token prediction.

    Args:
        logits: Model output, shape (batch, pos, vocab_size).
        tokens: Input tokens, shape (batch, pos).

    Returns:
        Loss per prediction, shape (batch, pos - 1). Entry i is the loss of
        predicting token i + 1 from tokens 0..i.
    """
    pred = logits[:, :-1]    # The last position has no next token to predict
    target = tokens[:, 1:]   # Token i + 1 is the answer for position i
    return F.cross_entropy(pred.transpose(1, 2), target, reduction="none")


def train(
    cfg: Config,
    seed: int = 0,
    steps: int = 15000,
    batch_size: int = 64,
    lr: float = 1e-3,
    half_len_range: tuple[int, int] = (20, 20),
    verbose: bool = True,
):
    """Train a Transformer on the induction task.

    Args:
        cfg: Model configuration.
        seed: Seed for weight initialization and data generation.
        steps: Number of optimizer steps.
        batch_size: Sequences per step.
        lr: Learning rate.
        half_len_range: Smallest and largest repeat length; one length is sampled uniformly per step. (20, 20) means a fixed length.
        verbose : Whether to print progress every 100 steps.

    Returns:
        The trained model and a list with the logged losses.
    """
    set_seed(seed)
    model = Transformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    #half_len = cfg.n_ctx // 2
    low, high = half_len_range
    assert 2 * high <= cfg.n_ctx, "longest sequence must fit in the context"
    history = []

    for step in range(steps):
        half_len = random.randint(low, high)
        tokens = make_batch(batch_size, half_len, cfg.vocab_size)
        logits = model(tokens)
        losses = per_position_loss(logits, tokens)
        loss = losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            first = losses[:, : half_len - 1].mean().item()
            second = losses[:, half_len:].mean().item()
            history.append(
                {"step": step, "half_len": half_len, "first_half": first, "second_half": second}
            )
            if verbose:
                print(
                    f"step {step:5d} | len {half_len:2d} | "
                    f"first half {first:.3f} | second half {second:.3f}"
                )

    return model, history


if __name__ == "__main__":
    model, history = train(Config(), steps=15000, half_len_range=(5, 20))
    torch.save(model.state_dict(), "part1_toy/varlen_seed0.pt")
