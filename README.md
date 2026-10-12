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
- [x] Synthetic task: repeated random token sequences (induction)
- [x] Tiny attention-only transformer (2 layers)
- [x] Training loop
- [x] Check the model learned the task (loss, accuracy on the repeated half)
- [x] Test on other repeat lengths; train a second model with variable repeat length
- [x] Hook points; read activations (cache)
- [x] Attention patterns and offset profile; verify the heads (induction in the variable model, fixed offset in the fixed model)
- [x] Write activations through hooks: zero ablation and mean ablation of single heads and groups
- [x] Direct logit attribution (residual stream decomposition)
- [x] Activation patching (clean vs corrupted prompt)
- [x] Repeat training with 5 seeds per training distribution and compare circuits (mechanism type is stable; division of labor varies with the seed)

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

Tested on Windows 11 with Python 3.13.5 and CPU-only PyTorch (no NVIDIA GPU).
Part 1 (tiny model) runs fine on CPU.

1. Create the virtual environment (from the repo root):

   ```bat
   py -3.13 -m venv .venv
   ```

   Use whichever Python 3.x version you have; `py -0` lists them.

2. Activate it. The command depends on your terminal:

   ```bat
   :: Windows cmd.exe
   .venv\Scripts\activate.bat
   ```
   ```powershell
   # Windows PowerShell (may need: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
   .venv\Scripts\Activate.ps1
   ```
   ```bash
   # Linux / macOS
   source .venv/bin/activate
   ```

   The prompt should start with `(.venv)`.

3. Upgrade pip and install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   This installs the CPU build of PyTorch. With an NVIDIA GPU, install the
   matching CUDA build first using the selector at
   https://pytorch.org/get-started/locally/, then run the last command.

4. Check the installation:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   Expected on this setup: `2.14.1+cpu False`.

5. VS Code only: select the venv interpreter (`Ctrl+Shift+P`, then
   "Python: Select Interpreter", then the `.venv` entry). Otherwise the
   editor reports unresolved imports even though they are installed.

## Running Part 1

Scripts in `part1_toy/` import each other directly (`from model import ...`),
so run them from that folder or point Python at the file:

```bash
python part1_toy/train.py     # trains the toy model, logs first/second-half loss
python part1_toy/test.py      # sanity checks: shapes, parameter count, initial loss
```

## Status

Part 1: toy models trained and verified (induction heads in the variable-length model, positional shortcut in the fixed-length one); single-head and group ablation (zero and mean) done, direct logit attribution done (layer 1 cancels layer 0's logit pattern in the first half), activation patching (denoising and noising) done; five-seed study per training distribution done (mechanism follows the training distribution, circuit details vary with the seed); optional next steps: weight-level check of the layer-1 cancellation, then Part 2.
