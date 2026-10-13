"""Probe: weight accessors and exactness of the GPT-2 logit decomposition."""

import torch
from transformer_lens.model_bridge import TransformerBridge

model = TransformerBridge.boot_transformers("gpt2")
model.enable_compatibility_mode()
tokens = model.to_tokens("When Mary and John went to the store, John gave a drink to")
with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)

for name in ["W_E", "W_pos", "W_O", "b_O", "W_U", "b_U"]:
    w = getattr(model, name, None)
    print(name, None if w is None else tuple(w.shape))

resid = cache["blocks.11.hook_resid_post"]
parts = cache["hook_embed"] + cache["hook_pos_embed"]
for layer in range(12):
    parts = parts + cache[f"blocks.{layer}.hook_attn_out"] + cache[f"blocks.{layer}.hook_mlp_out"]
print("components sum to final residual:", (parts - resid).abs().max().item())

scale = cache["ln_final.hook_scale"]
normed = cache["ln_final.hook_normalized"]
print("residual mean (centered?):", resid.mean(-1).abs().max().item())
print("normed vs (resid - mean)/scale:", (normed - (resid - resid.mean(-1, keepdim=True)) / scale).abs().max().item())
b_U = model.b_U if getattr(model, "b_U", None) is not None else 0
rebuilt = normed @ model.W_U + b_U
print("logits vs normed @ W_U + b_U:", (rebuilt - logits).abs().max().item())
