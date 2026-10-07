"""Check that hook-based extraction equals HF output_hidden_states at the last token.

    uv run python scripts/check_hooks.py <model>
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tdnv.extract import last_token_states
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto").eval()
texts = ["The city of Krasnodar is in Russia.", "Hi", "Seventy-six is larger than fifty-five, said the long sentence here."]
new = last_token_states(model, tok, texts, batch_size=3, max_length=64)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
with torch.no_grad():
    hs = model(**enc, output_hidden_states=True).hidden_states
old = torch.stack([h[:, -1].float().cpu() for h in hs], 1)
print(name, tuple(new.shape), tuple(old.shape), "max abs diff", (new - old).abs().max().item(),
      "max rel", ((new - old).norm(dim=-1) / old.norm(dim=-1).clamp_min(1e-6)).max().item())
