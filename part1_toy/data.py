"""Synthetic induction task: random tokens repeated twice.

We want a task where we know which mechanism a model needs, so we can check
whether interventions find it. The classic choice is induction:

    If the text contains [A][B] ... [A], then after the second [A] predict [B].

Each sequence is a random half repeated twice. The tokens are fresh random
draws, so memorization and token statistics cannot help. The only way to lower
the loss on the second half is to look back in the context.

Hypothesis: the model learns to look back and find what followed the current
token last time, and interventions can locate the components responsible.
Caveat: a positional shortcut (copy from `half_len` positions ago) would also
work, so this has to be tested.
"""

import torch


def make_batch(batch_size: int, half_len: int, vocab_size: int) -> torch.Tensor:
    """Generate a batch of sequences made of a random half repeated twice.

    Tokens within one sequence are distinct (sampled without replacement), so
    the second-half prediction is never ambiguous.
    Requires vocab_size >= half_len.

    Args:
        batch_size: Number of sequences.
        half_len: Length of the random half.
        vocab_size: Number of distinct token ids (0 .. vocab_size - 1).

    Returns:
        Integer tensor of shape (batch_size, 2 * half_len).
    """
    # A random permutation of the vocabulary per row; keep the first half_len ids
    first = torch.rand(batch_size, vocab_size).argsort(dim=1)[:, :half_len]
    # Repeat the half along the sequence dimension
    return torch.cat([first, first], dim=1)
