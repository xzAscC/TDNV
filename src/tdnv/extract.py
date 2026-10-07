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


def _decoder_stack(model):
    """Return (decoder layers, final norm) of a HF causal LM, including multimodal Gemma-3."""
    layers = max(
        (m for m in model.modules() if isinstance(m, torch.nn.ModuleList)
         and len(m) and type(m[0]).__name__.endswith("DecoderLayer")),
        key=len,
    )
    owner = next(m for m in model.modules() if any(c is layers for c in m.children()))
    return layers, owner.norm


@torch.no_grad()
def last_token_states(model, tokenizer, texts: list[str], batch_size: int, max_length: int):
    """Return float32 CPU tensor [N, L+1, d] at the last token, matching HF `hidden_states`.

    Index 0 is the embedding output, index l the output of decoder layer l, and the last index
    is after the final norm. Hooks keep only the last position of each layer, so memory does
    not grow with sequence length and large batches fit; this matters when a model is split
    over GPUs without peer-to-peer links, where each forward pass has a fixed cross-GPU cost.
    """
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    layers, norm = _decoder_stack(model)
    captured: list[torch.Tensor] = []

    def grab_input(_mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        captured.append(h[:, -1].to("cpu", torch.float32, non_blocking=False))

    handles = [l.register_forward_pre_hook(grab_input, with_kwargs=True) for l in layers]
    handles.append(norm.register_forward_hook(
        lambda _m, _i, out: captured.append(out[:, -1].to("cpu", torch.float32))))
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucketed batches
    out = []
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="extract", mininterval=10):
            enc = tokenizer(
                [texts[j] for j in order[i : i + batch_size]], return_tensors="pt", padding=True,
                truncation=True, max_length=max_length, add_special_tokens=False,
            ).to(model.device)
            captured.clear()
            model(**enc, logits_to_keep=1)
            # pre-hooks give [emb, out_0, ..., out_{L-2}]; the norm hook gives norm(out_{L-1})
            out.append(torch.stack(captured, 1))
    finally:
        for h in handles:
            h.remove()
    sorted_hidden = torch.cat(out)
    hidden = torch.empty_like(sorted_hidden)
    hidden[torch.tensor(order)] = sorted_hidden
    return hidden
