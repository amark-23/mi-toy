"""Shared helpers for Part 1: reproducible seeding."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator used in this project.

    The same seed gives the same synthetic data, initial weights and batch
    order, so runs are repeatable. The seed is also an experimental variable:
    different seeds can lead to different circuits for the same behavior.

    Args:
        seed: Integer seed shared by all generators.
    """
    random.seed(seed)         # Python's built-in RNG
    np.random.seed(seed)      # NumPy global RNG
    torch.manual_seed(seed)   # PyTorch RNG: weight init and torch.rand in the data
