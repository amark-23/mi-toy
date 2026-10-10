# mi-toy: a mechanistic interpretability warmup

A short, hands-on project to learn the core toolkit of mechanistic
interpretability (MI) before moving to open research questions.
Nothing here is novel research. The goal is to understand the mechanics
and to learn what a trustworthy result looks like.

Theory notes, remarks and open questions live in [`info.md`](info.md).

## Goals

1. Build a tiny transformer from scratch and train it on a synthetic task.
2. Write the interventions by hand (hooks, ablation, activation patching,
   direct logit attribution) so the mechanics are fully visible.
3. Repeat the toolkit on a pretrained model (GPT-2 small, IOI task from
   Wang et al., 2023, "Interpretability in the Wild") with a library.
4. Note the pain points. They may become the research question.

## Structure

```
mi-toy/
├── README.md          main context and progress (this file)
├── info.md            theory notes, remarks, open questions
├── requirements.txt   dependencies (unpinned until first install)
├── part1_toy/         toy model + hand-written interventions
├── part2_gpt2_ioi/    GPT-2 small + IOI with a library
└── figures/           saved plots used in the notes
```

## Roadmap

### Part 1: toy model, plain PyTorch
- [ ] Synthetic task: repeated random token sequences (induction)
- [ ] Tiny attention-only transformer (1-2 layers)
- [ ] Training loop
- [ ] Check the model learned the task (loss, accuracy on the repeated half)
- [ ] Forward hooks to read and write activations
- [ ] Direct logit attribution (residual stream decomposition)
- [ ] Zero ablation and mean ablation of single heads
- [ ] Activation patching (clean vs corrupted prompt)
- [ ] Verify the discovered heads show the induction attention pattern
- [ ] Optional: repeat training with 3-5 seeds and compare which heads matter

### Part 2: GPT-2 small + IOI
- [ ] Set up the library, load the model
- [ ] IOI prompts, metric (logit difference), two corruption schemes
- [ ] Logit lens and direct logit attribution
- [ ] Activation patching by layer and head
- [ ] Path patching for one edge
- [ ] Attribution patching vs exact patching
- [ ] Faithfulness number for the circuit found
- [ ] Optional: one pretrained SAE feature (Gemma Scope) and steering

## Conventions

- Fixed seeds everywhere. One notebook (or script) per result.
- Define the metric before looking at results, and justify the corruption.
- Report numbers, not only pictures.
- Every change to the repo is explained when it is made (see `info.md`).

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Status

Scaffold created. No code yet.
