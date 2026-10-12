"""Train several seeds of each model kind for the circuit-stability study."""

import os

import torch

from model import Config
from train import train

KINDS = {"fixed": (20, 20), "varlen": (5, 20)}
SEEDS = [0, 1, 2, 3, 4]
STEPS = 15000

if __name__ == "__main__":
    cfg = Config()
    os.makedirs("part1_toy/checkpoints", exist_ok=True)
    for kind, half_len_range in KINDS.items():
        for seed in SEEDS:
            path = f"part1_toy/checkpoints/{kind}_seed{seed}.pt"
            if os.path.exists(path):
                print(f"skip {path} (exists)")
                continue
            model, history = train(
                cfg, seed=seed, steps=STEPS, half_len_range=half_len_range, verbose=False
            )
            torch.save(model.state_dict(), path)
            last = history[-1]
            print(
                f"{kind} seed {seed}: step {last['step']} | "
                f"first half {last['first_half']:.3f} | second half {last['second_half']:.3f}"
            )