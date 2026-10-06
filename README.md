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

## Concept sweep (branch `sweep`)

`scripts/run_concepts.py` runs every concept in `src/tdnv/concepts.py` for one model; `slurm/concepts.sh`
submits it on OSC. Concepts: 7 CAA behaviors (Rimsky et al., last token = the answer letter), 5
Geometry-of-Truth true/false sets (Marks & Tegmark, last token = the final period), and the two safety
settings above. Up to 1000 examples per class.

`notebooks/plot_sweep.py` writes `results/u_shape_summary.csv` and the figures in `figs/`
(per-concept figures in `figs/concepts/`). A run counts as down-then-up when TDNV falls at least 2x
from its earlier peak to the minimum, rises at least 1.5x from the minimum to the last layer, and the
minimum is not in the last 10% of depth.

| Model | Layers | Down-then-up concepts (of 14) |
|---|---|---|
| Qwen3-4B / 8B / 14B / 32B | 36 / 36 / 40 / 64 | 6 / 7 / 10 / 11 |
| Gemma-2-9B / 27B | 42 / 46 | 10 / 9 |
| Gemma-3-12B / 27B | 48 / 62 | 13 / 14 |
| OLMo-3-7B / OLMo-3.1-32B | 32 / 64 | 8 / 11 |
| Llama-3.1-8B | 32 | 12 |

Myopic Reward and Corrigibility are down-then-up in all 11 models; Cities, Larger Than and Companies in
10 of 11. Survival Instinct is the exception (3 of 11).

### Running models split over two GPUs on OSC Ascend

On the 2-GPU A100 nodes, CUDA reports peer access but direct GPU-to-GPU copies are slow and corrupt
data, so a model split with `device_map="auto"` gives wrong hidden states (NaN for the 27-32B models).
`src/tdnv/p2p.py` routes those copies through the CPU; `scripts/check_split.py` checks a split model
against a single-GPU run.
