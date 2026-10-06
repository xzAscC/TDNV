"""Last-token hidden states at every layer."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm


def render(tokenizer, conv: list[dict], chat_template: bool, max_response_tokens: int) -> str:
    """Turn a conversation into the exact string whose last token we read.

    A user-only conversation ends with the generation prompt (the position where the model
    decides how to answer). For a conversation with an assistant turn, the response is cut to
    max_response_tokens and appended without an end-of-turn token, so both classes end on
    response text rather than on the same special token.
    """
    user, resp = conv[0]["content"], conv[1]["content"] if len(conv) > 1 else None
    if chat_template and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = (tokenizer.bos_token or "") + (user if resp is None else user + "\n")
    if resp is not None:
        ids = tokenizer(resp, add_special_tokens=False)["input_ids"][:max_response_tokens]
        text += tokenizer.decode(ids)
    return text


@torch.no_grad()
def last_token_states(model, tokenizer, texts: list[str], batch_size: int, max_length: int):
    """Return float32 CPU tensor [N, L+1, d]; index 0 is the embedding output."""
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="extract"):
        enc = tokenizer(
            texts[i : i + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length, add_special_tokens=False,
        ).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1] for h in hs], 1).float().cpu())
    return torch.cat(out)
