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

_empty_

## 3. Part 2 notes (GPT-2 small, IOI)

_empty_

## 4. Remarks and surprises

Things that behaved differently than expected. These are candidate
research questions.

_empty_

## 5. Change log

Every change to the repo, with the reason.

- Scaffold created: folders, README, info.md, requirements.txt, .gitignore.
