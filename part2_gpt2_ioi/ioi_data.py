"""IOI prompt dataset: templates, name pairs, clean and corrupted versions."""

import random

import torch

NAMES = ["Mary", "John", "Tom", "James", "Dan", "Sarah", "Alice", "Ben", "Kate", "Paul", "Anna", "Mark"]
PLACES = ["store", "park", "school", "office", "beach", "station"]
OBJECTS = ["drink", "book", "letter", "gift", "ring", "bone"]

TEMPLATES = [
    "When {first} and {second} went to the {place}, {S} gave a {object} to",
    "Then, {first} and {second} had a lot of fun at the {place}. {S} gave a {object} to",
    "After {first} and {second} went to the {place}, {S} gave a {object} to",
]


def single_token_names(model, names):
    keep = []
    for name in names:
        ids = model.to_tokens(" " + name, prepend_bos=False)
        if ids.shape[1] == 1:
            keep.append(name)
    return keep


def make_prompts(n, names, seed=0):
    """Return n IOI prompts as dicts.

    Keys: text (clean), corrupt (ABC version: the final subject replaced by a third
    name C), IO, S, C, order (ABBA or BABA), template index.
    """
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        io, s, c = rng.sample(names, 3)
        order = rng.choice(["ABBA", "BABA"])
        first, second = (io, s) if order == "ABBA" else (s, io)
        t = rng.randrange(len(TEMPLATES))
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        fields = dict(first=first, second=second, place=place, object=obj)
        text = TEMPLATES[t].format(S=s, **fields)
        corrupt = TEMPLATES[t].format(S=c, **fields)
        prompts.append(
            {"text": text, "corrupt": corrupt, "IO": io, "S": s, "C": c, "order": order, "template": t}
        )
    return prompts


def lengths_by_template(model, prompts):
    """Map template index to the set of token lengths seen (should be one value each)."""
    out = {}
    for p in prompts:
        n = model.to_tokens(p["text"]).shape[1]
        out.setdefault(p["template"], set()).add(n)
    return out


def build_batch(model, prompts, template):
    """Token batches for one template (all rows have the same length)."""
    chosen = [p for p in prompts if p["template"] == template]
    tokens = torch.cat([model.to_tokens(p["text"]) for p in chosen])
    corrupt = torch.cat([model.to_tokens(p["corrupt"]) for p in chosen])
    io = torch.tensor([model.to_single_token(" " + p["IO"]) for p in chosen])
    s = torch.tensor([model.to_single_token(" " + p["S"]) for p in chosen])
    abba = torch.tensor([p["order"] == "ABBA" for p in chosen])
    return {"tokens": tokens, "corrupt": corrupt, "io": io, "s": s, "abba": abba}


def logit_diff(logits, io, s):
    """logit(IO) - logit(S) at the last position, one value per prompt."""
    last = logits[:, -1]
    rows = torch.arange(last.shape[0])
    return last[rows, io] - last[rows, s]


if __name__ == "__main__":
    from transformer_lens.model_bridge import TransformerBridge

    model = TransformerBridge.boot_transformers("gpt2")
    model.enable_compatibility_mode()
    print(single_token_names(model, NAMES))
    print(lengths_by_template(model, make_prompts(500, NAMES, seed=0)))
    prompts = make_prompts(900, NAMES, seed=0)
    print("template | prompts | order | clean LD (frac>0) | corrupt LD (frac>0)")
    for t in range(len(TEMPLATES)):
        batch = build_batch(model, prompts, t)
        with torch.no_grad():
            clean_ld = logit_diff(model.run_with_cache(batch["tokens"])[0], batch["io"], batch["s"])
            corrupt_ld = logit_diff(model.run_with_cache(batch["corrupt"])[0], batch["io"], batch["s"])
        for name, mask in (("ABBA", batch["abba"]), ("BABA", ~batch["abba"])):
            c, k = clean_ld[mask], corrupt_ld[mask]
            print(
                f"{t:8d} | {len(c):7d} | {name} | {c.mean():6.2f} ({(c > 0).float().mean():.2f})"
                f" | {k.mean():6.2f} ({(k > 0).float().mean():.2f})"
            )
