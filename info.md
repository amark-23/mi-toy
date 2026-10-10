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

_empty_

## 5. Change log

Every change to the repo, with the reason.

- Scaffold created: folders, README, info.md, requirements.txt, .gitignore.
- Environment set up (venv with Python 3.13.5, CPU-only PyTorch 2.14.1). Setup steps documented in README.md so the repo is reproducible. Pinning exact versions in requirements.txt is still pending.
- `part1_toy/utils.py` started by the user (imports for seeding).
- `part1_toy/data.py` added by the user (`make_batch`). Added Part 1 task description, hypothesis and the positional-shortcut caveat to info.md section 2.
- `part1_toy/model.py` started (`Config`, `Embed`, `PosEmbed`, `Attention` parameters). `make_batch` changed to sample without replacement, because duplicate tokens within a sequence make the second-half prediction ambiguous (birthday problem).
- Comment style standardized in `utils.py`, `data.py`, `model.py`: module docstring at the top, Google-style docstrings (Args/Returns, tensor shapes in the form `(batch, pos, d_model)`), short inline comments for single lines, two blank lines between top-level definitions. Wording corrected where it was inaccurate (bias terms, "force", residual stream width).
- `Config` was missing `d_head = 16`, which `Attention` needs; added.
