"""Shared figure style for the TDNV paper figures (house style: Source Sans, print size)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

FONT_DIR = Path("/usr/share/texmf-dist/fonts/opentype/adobe/sourcesanspro")
for f in FONT_DIR.glob("SourceSansPro-*.otf"):
    font_manager.fontManager.addfont(str(f))

INK, MUTED, RULE = "#203342", "#74818A", "#DCE3E6"
FULL, HALF = 5.5, 2.7  # inches: \textwidth and half width

RC = {
    "font.family": "Source Sans Pro", "mathtext.fontset": "custom",
    "mathtext.rm": "Source Sans Pro", "mathtext.it": "Source Sans Pro:italic",
    "mathtext.bf": "Source Sans Pro:bold", "mathtext.cal": "cmsy10",
    "mathtext.fallback": "stixsans", "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": RULE, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "ytick.minor.size": 0, "lines.linewidth": 1.0, "lines.markersize": 3,
    "legend.frameon": False, "figure.dpi": 150,
}
mpl.rcParams.update(RC)

# Color encodes the model family; marker shape encodes the model within its family.
FAMILY_COLOR = {"Qwen3": "#3A6DB5", "Gemma-2": "#D04A4A", "Gemma-3": "#3E8E3E", "OLMo-3": "#B88A12"}
MODELS = {  # hf id -> (family, short label, marker)
    "Qwen/Qwen3-4B": ("Qwen3", "Qwen3-4B", "v"),
    "Qwen/Qwen3-8B": ("Qwen3", "Qwen3-8B", "o"),
    "Qwen/Qwen3-14B": ("Qwen3", "Qwen3-14B", "s"),
    "Qwen/Qwen3-32B": ("Qwen3", "Qwen3-32B", "^"),
    "google/gemma-2-9b-it": ("Gemma-2", "Gemma-2-9B", "o"),
    "google/gemma-2-27b-it": ("Gemma-2", "Gemma-2-27B", "s"),
    "google/gemma-3-12b-it": ("Gemma-3", "Gemma-3-12B", "^"),
    "google/gemma-3-27b-it": ("Gemma-3", "Gemma-3-27B", "D"),
    "allenai/Olmo-3-7B-Instruct": ("OLMo-3", "OLMo-3-7B", "o"),
    "allenai/Olmo-3-1125-32B": ("OLMo-3", "OLMo-3-32B", "s"),  # base model: no 3.0 32B Instruct
}

# Concepts in the paper: Cities (main text) and six more (appendix), in panel order.
CONCEPT_LABEL = {
    "truth_cities": "Truth: Cities",
    "caa_corrigible-neutral-HHH": "Corrigibility", "caa_myopic-reward": "Myopic Reward",
    "truth_sp_en_trans": "Truth: Translation", "truth_larger_than": "Truth: Larger Than",
    "truth_companies_true_false": "Truth: Companies", "truth_common_claim_true_false": "Truth: Common Claims",
}


def model_style(model_id: str) -> dict:
    fam, label, marker = MODELS[model_id]
    return dict(color=FAMILY_COLOR[fam], marker=marker, label=label)
