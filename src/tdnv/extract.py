"""Last-token hidden states at every layer."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm


def render(tokenizer, ex, max_response_tokens: int = 32) -> str:
    """Turn an Example into the exact string whose last token we read.

    A chat example without an assistant turn ends with the generation prompt (where the model
    decides how to answer). With an assistant turn, the response is cut to max_response_tokens
    and appended without an end-of-turn token, so the last token is response text.
    A raw example is the plain text with a BOS token.
    """
    if ex.raw or not tokenizer.chat_template:
        text = (tokenizer.bos_token or "") + ex.user
        if ex.assistant is not None:
            text += "\n" + ex.assistant
        return text
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": ex.user}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    if ex.assistant is not None:
        ids = tokenizer(ex.assistant, add_special_tokens=False)["input_ids"][:max_response_tokens]
        text += tokenizer.decode(ids)
    return text


@torch.no_grad()
def last_token_states(model, tokenizer, texts: list[str], batch_size: int, max_length: int):
    """Return float32 CPU tensor [N, L+1, d]; index 0 is the embedding output."""
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucketed batches
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="extract", mininterval=10):
        enc = tokenizer(
            [texts[j] for j in order[i : i + batch_size]], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length, add_special_tokens=False,
        ).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1].to("cpu", torch.float32) for h in hs], 1))
    sorted_hidden = torch.cat(out)
    hidden = torch.empty_like(sorted_hidden)
    hidden[torch.tensor(order)] = sorted_hidden
    return hidden
