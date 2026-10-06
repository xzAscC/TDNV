# TDNV on contrastive (steering) data

Layerwise **Task-Distance Normalized Variance** of last-token hidden states, as in
*From Compression to Expansion: A Layerwise Analysis of In-Context Learning*
([arXiv 2505.17322](https://arxiv.org/abs/2505.17322)), applied to positive/negative datasets
that are used for activation steering, instead of toy ICL tasks.

```
TDNV^(l) = sum_{t != t'} (var_t + var_t') / (2 ||mu_t - mu_t'||^2)
```

For two classes this is `(var_0 + var_1) / ||mu_0 - mu_1||^2`. A value below 1 means the two class
means are farther apart than the spread inside each class.

## Safety setting

Data: [`LLM-LAT/harmful-dataset`](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) and
[`LLM-LAT/benign-dataset`](https://huggingface.co/datasets/LLM-LAT/benign-dataset).

| mode | label 1 | label 0 | last token |
|---|---|---|---|
| `prompt` | harmful prompt | benign prompt | end of the chat generation prompt |
| `response` | harmful prompt + compliant answer (`rejected`) | same prompt + refusal (`chosen`) | token `--max-response-tokens` of the answer |

For each layer the script reports:

- `tdnv`: TDNV with the true labels.
- `tdnv_shuffled`: mean TDNV over random label permutations (the no-structure baseline).
- `within_var`, `between_dist2`: the numerator and denominator.
- `probe_acc`: held-out accuracy of the difference-of-means direction (the usual steering vector),
  fit on half of each class.

Layer 0 is the embedding output. In `prompt` mode every input ends on the same template token,
so layer 0 has zero variance and zero distance, and its TDNV is `NaN`.

## Run

```bash
uv sync
uv run pytest -q
uv run python scripts/run_safety.py --model meta-llama/Llama-3.1-8B-Instruct --mode prompt --n 1000
```

On OSC Ascend (one A100):

```bash
sbatch slurm/safety.sh
MODELS="Qwen/Qwen3-8B" N=2000 sbatch slurm/safety.sh
```

Outputs go to `outputs/<model>/<mode>_<chat|raw>_n<N>_s<seed>/` (`metrics.json`, `tdnv.png`).
