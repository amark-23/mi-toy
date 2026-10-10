# info.md: theory notes and remarks

Working notes for `mi-toy`. Each section is filled in as we reach it.

## 1. Glossary

| Term | Short meaning |
|---|---|
| Residual stream | Shared vector per token position that every layer reads from and adds to |
| Attention head | Reads the stream, decides where to attend (QK) and what to move (OV), writes back |
| Logit | Pre-softmax score for a vocabulary token |
| Direct logit attribution (DLA) | Contribution of one component to the final logits through the residual stream |
| Ablation | Remove a component's contribution (zero or mean value) and observe the change |
| Activation patching | Copy an activation from one run into another and observe the change |
| Clean / corrupted prompt | The two inputs used in patching; differ in the minimal detail that matters |
| Faithfulness | The found circuit alone reproduces the model's behavior |
| Superposition | More features than dimensions, stored as almost-orthogonal directions |

(to be extended)

## 2. Part 1 notes (toy model)

**Task (induction on random tokens).** Each sequence is a random half repeated twice, e.g. `K R P Z K R P Z`. The model predicts the next token at every position. In the first half the tokens are unpredictable (loss at chance, ln(vocab)). In the second half the answer exists earlier in the input, so the only way to improve is to look back: "last time I saw this token, this one followed it". Random tokens are used on purpose, so that language knowledge or memorization cannot help.

**Hypothesis.** The model will learn this look-back behavior by itself (nothing in the code encodes it), and interventions should locate the components responsible. This is a hypothesis to test, not a guarantee.

**Caveat: copying is not necessarily induction.** With a fixed half length, a positional shortcut ("predict the token from `half_len` positions ago") also scores perfectly. Planned check: test on repeat lengths unseen in training, and inspect whether the found heads attend to the token *after* the earlier occurrence.

## 3. Part 2 notes (GPT-2 small, IOI)

_empty_

## 4. Remarks and surprises

Things that behaved differently than expected. These are candidate
research questions.

**Fixed-length training gave a positional shortcut, not induction (seed 0, 10,000 steps).**
Trained with a fixed repeat length of 20, the model reaches second-half loss 0.033 (first half about 4.65, i.e. chance). Test on other repeat lengths (10 batches of 256 sequences each):

| Repeat length | First half | Second half |
|---|---|---|
| 5 | 4.660 | 4.697 |
| 10 | 4.671 | 4.641 |
| 15 | 4.668 | 61.162 |
| 20 (trained) | 4.661 | 0.059 |

Criteria fixed before running: second half below ~0.5 means content-based, above ~3 means positional, in between means mixed. All shifted lengths are above 3, so the model learned a positional rule ("predict the token 20 positions back"). It fires only at positions of 19 or more, which explains chance loss for lengths 5 and 10 (no position reaches 19) and the huge loss for length 15 (positions 20-28 confidently copy from the wrong distance; logits are unbounded because there is no normalization).
This is inferred from losses only; it still has to be confirmed by inspecting attention patterns.
Link to the research theme: equal accuracy, different mechanism.
Next: train a second model with the repeat length sampled per step, so only content-based lookup works, and compare the two models.

**Variable-length model (seed 0, 15,000 steps, repeat length sampled from 5 to 20 each step).**
Same test, 10 batches of 256 sequences, both models with the same test data:

| Repeat length | Fixed model, second half | Variable model, second half |
|---|---|---|
| 5 | 4.697 | 0.135 |
| 10 | 4.641 | 0.058 |
| 15 | 61.162 | 0.024 |
| 20 | 0.059 | 0.032 |

Supported claim: the variable model copies by content, not by a fixed distance (one set of weights works across lengths, which a fixed-offset rule cannot do).
Not yet supported: that it implements induction (previous-token head plus induction head). All tested lengths were inside the training range, and behavior alone cannot rule out other content-based algorithms (e.g. detect the first token reappearing, then copy at that offset).
Needed to claim induction: attention patterns of the layer-0 and layer-1 heads, head ablations, and a harder behavioral test (repeated block starting mid-sequence, or pairs reappearing in shuffled order).
Note: this run used 15,000 steps, the fixed-length run 10,000.

## 5. Change log

Every change to the repo, with the reason.

- Scaffold created: folders, README, info.md, requirements.txt, .gitignore.
- Environment set up (venv with Python 3.13.5, CPU-only PyTorch 2.14.1). Setup steps documented in README.md so the repo is reproducible. Pinning exact versions in requirements.txt is still pending.
- `part1_toy/utils.py` started by the user (imports for seeding).
- `part1_toy/data.py` added by the user (`make_batch`). Added Part 1 task description, hypothesis and the positional-shortcut caveat to info.md section 2.
- `part1_toy/model.py` started (`Config`, `Embed`, `PosEmbed`, `Attention` parameters). `make_batch` changed to sample without replacement, because duplicate tokens within a sequence make the second-half prediction ambiguous (birthday problem).
- Comment style standardized in `utils.py`, `data.py`, `model.py`: module docstring at the top, Google-style docstrings (Args/Returns, tensor shapes in the form `(batch, pos, d_model)`), short inline comments for single lines, two blank lines between top-level definitions. Wording corrected where it was inaccurate (bias terms, "force", residual stream width).
- `Config` was missing `d_head = 16`, which `Attention` needs; added.
- `part1_toy/train.py` added (`per_position_loss`, `train`). Entry point `if __name__ == "__main__": train(Config())` added at the bottom so it can be run directly.
- Imports inside `part1_toy/` changed from `from part1_toy.x import ...` to sibling imports (`from x import ...`) so the scripts run directly from that folder. Consequence: they can no longer be imported from the repo root as `part1_toy.train`. Scratch checks moved from root `test.py` to `part1_toy/test.py` (same content, updated imports). README has a new "Running Part 1" section.
- `part1_toy/evaluate.py` added (`load_model`, `eval_loss`) and `train.py` now saves `part1_toy/seed0.pt` (ignored by git via `*.pt`). First result and finding recorded in info.md section 4.
- `train.py` gained `half_len_range` (repeat length sampled per step); `evaluate.py` now compares both checkpoints (`seed0.pt` fixed length, `varlen_seed0.pt` variable length). Results and the calibrated claim recorded in info.md section 4.
