"""First look at GPT-2 small through TransformerBridge: tokens, logits, cache keys."""

from transformer_lens.model_bridge import TransformerBridge

model = TransformerBridge.boot_transformers("gpt2")
model.enable_compatibility_mode()

prompt = "When Mary and John went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)
print(tokens.shape)
print(model.to_str_tokens(prompt))

logits, cache = model.run_with_cache(tokens)
print(logits.shape)
for key in list(cache.keys())[:30]:
    print(key, tuple(cache[key].shape))

print([k for k in cache.keys() if not k.startswith("blocks.")])
print([k for k in cache.keys() if k.startswith("blocks.0.") and "attn" not in k])


mary = model.to_single_token(" Mary")
john = model.to_single_token(" John")
last = logits[0, -1]
print(last[mary].item(), last[john].item(), (last[mary] - last[john]).item())
print(model.to_str_tokens(last.topk(5).indices))