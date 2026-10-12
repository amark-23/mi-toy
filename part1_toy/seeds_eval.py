"""Evaluate every trained seed: did it learn the task, and does it generalize?"""

import torch

from evaluate import eval_loss, load_model
from model import Config
from utils import set_seed

KINDS = ["fixed", "varlen"]
SEEDS = [0, 1, 2, 3, 4]
LENGTHS = [5, 10, 15, 20]

if __name__ == "__main__":
    cfg = Config()

    for kind in KINDS:
        print(f"\n{kind}: second-half loss by repeat length")
        print("  seed |" + "".join(f" len {n:2d} |" for n in LENGTHS))
        for seed in SEEDS:
            model = load_model(f"part1_toy/checkpoints/{kind}_seed{seed}.pt", cfg)
            cells = []
            for half_len in LENGTHS:
                set_seed(0)
                _, second = eval_loss(model, half_len, cfg.vocab_size)
                cells.append(second)
            print(f"  {seed:4d} |" + "".join(f" {c:7.3f} |" for c in cells))

    old = torch.load("part1_toy/varlen_seed0.pt")
    new = torch.load("part1_toy/checkpoints/varlen_seed0.pt")
    same = all(torch.equal(old[key], new[key]) for key in old)
    print("\nvarlen seed 0 reproduces the earlier checkpoint exactly:", same)

    for key in old:
        diff = (old[key] - new[key]).abs().max().item()
        print(f"  {key:28s} max |old - new| = {diff:.2e}")    