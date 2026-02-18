#!/usr/bin/env python3
"""
Build the CultureSense Kaggle notebook (culturesense.ipynb)
by reading the source .py files and assembling notebook cells.

All source files are inlined as notebook cells that share one Python kernel.
Inter-module imports (e.g. "from data_models import ...") are stripped because
every symbol is already defined in a prior cell of the same kernel namespace.
"""

import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Local module names — any import of these is stripped when inlining
# ---------------------------------------------------------------------------
LOCAL_MODULES = {
    "data_models",
    "rules",
    "extraction",
    "trend",
    "hypothesis",
    "medgemma",
    "renderer",
    "evaluation",
    "extraction_agent",
}


def read_src(filename: str) -> str:
    with open(os.path.join(BASE, filename)) as f:
        return f.read()


def strip_local_imports(source: str) -> str:
    """
    Remove any import line that references a local module.

    Handles these forms:
        from data_models import Foo, Bar
        import data_models
        from rules import RULES, normalize_organism  (including multi-line continuations)

    Also strips the module-level docstring (triple-quoted string at the top)
    to avoid duplication across cells.
    """
    lines = source.splitlines(keepends=True)
    result = []
    skip_continuation = False

    # Strip leading module docstring ("""...""" or '''...''')
    # Walk lines until we find and consume one triple-quoted block at the top.
    i = 0
    # Skip any leading blank lines / shebang
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    if i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            rest_of_line = stripped[3:]
            if quote in rest_of_line:
                # Docstring opens and closes on the same line
                i += 1
            else:
                # Multi-line docstring: advance until we find the closing quote
                i += 1
                while i < len(lines):
                    if quote in lines[i]:
                        i += 1
                        break
                    i += 1
        # else: no leading docstring — leave i unchanged, process from here

    remaining_lines = lines[i:]

    for line in remaining_lines:
        # Handle backslash continuations from a previous stripped import
        if skip_continuation:
            if not line.rstrip().endswith("\\"):
                skip_continuation = False
            continue

        # Match "from <local_module> import ..." or "import <local_module>"
        m_from = re.match(r"^\s*from\s+(\w+)\s+import\b", line)
        m_import = re.match(r"^\s*import\s+(\w+)\b", line)

        if m_from and m_from.group(1) in LOCAL_MODULES:
            if line.rstrip().endswith("\\"):
                skip_continuation = True
            continue  # drop this line

        if m_import and m_import.group(1) in LOCAL_MODULES:
            if line.rstrip().endswith("\\"):
                skip_continuation = True
            continue  # drop this line

        result.append(line)

    # Strip trailing blank lines
    source_out = "".join(result).rstrip("\n")
    return source_out


def inline(filename: str) -> str:
    """Read a source file and strip its local inter-module imports."""
    return strip_local_imports(read_src(filename))


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ============================================================
# Static cell content
# ============================================================

TITLE_MD = """# CultureSense — Longitudinal Clinical Hypothesis Engine
## Kaggle HAI-DEF Competition Submission

[![MedGemma](https://img.shields.io/badge/MedGemma-4b--it-blue)](https://huggingface.co/google/medgemma-4b-it)
[![Safety](https://img.shields.io/badge/Safety-Non--Diagnostic-green)]()
[![Mode](https://img.shields.io/badge/Mode-Patient%20%2B%20Clinician-purple)]()

> **CultureSense** processes 2–3 sequential urine or stool culture lab reports and produces
> structured, **non-diagnostic** interpretations through two distinct output modes.
> MedGemma handles natural language generation from already-structured inputs.
> Deterministic rules handle all temporal signal extraction.

---

## Architecture

```mermaid
flowchart TD
    A["[1] Raw Report Ingestion\\nList[str] 2-3 free-text culture reports"] --> B
    B["[2] Structured Extraction Layer\\nextract_structured_data() → CultureReport"] --> C
    C["[3] Temporal Comparison Engine\\nanalyze_trend() → TrendResult"] --> D
    D["[4] Hypothesis Update Layer\\ngenerate_hypothesis() → HypothesisResult\\nconfidence [0.0–0.95]"] --> E
    E["[5] MedGemma Reasoning Layer\\ncall_medgemma(structured_payload, mode) → str\\nModes: patient | clinician"] --> F
    F["[6] Structured Safe Output Renderer\\nrender_output() → FormattedOutput\\nPatient: explanation + questions\\nClinician: trajectory + confidence + flags"]

    style A fill:#e8f4f8
    style B fill:#d4edda
    style C fill:#d4edda
    style D fill:#fff3cd
    style E fill:#f8d7da
    style F fill:#e8f4f8
```

**Key safety invariant:** Raw report text is NEVER forwarded to MedGemma.
Only derived structured fields (typed dataclasses → JSON) are passed to the model.

---
"""

SETUP_MD = "## Cell A: Setup & Imports"

SETUP_CODE = """# Cell A-2: Library Installation
import subprocess, sys

packages = [
    "transformers>=4.40.0",
    "accelerate>=0.29.0",
    "sentencepiece>=0.1.99",
    "huggingface_hub>=0.22.0",
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Installation complete.")
"""

IMPORTS_CODE = """# Cell A-3: Core Imports
from __future__ import annotations
import re, json, warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("transformers not available — stub mode will be used.")

print("Imports complete.")
"""

MODELS_MD = "## Cell B: Data Models & Rule Library"
RULES_MD = "## Cell C: Extraction Layer"
TREND_MD = "## Cell D: Temporal Trend Engine"
HYPOTHESIS_MD = "## Cell E: Hypothesis Update Layer"
MEDGEMMA_MD = "## Cell F: MedGemma Integration"
RENDERER_MD = "## Cell G: Output Renderer"
DEMO_MD = """## Cell H: Demo Run

Three simulated scenarios demonstrate the full pipeline end-to-end.

| Scenario | Expected Trend | Expected Confidence |
|----------|---------------|---------------------|
| A — Improving Infection | decreasing | ≥ 0.80 |
| B — Emerging Resistance | fluctuating | < 0.80, stewardship alert |
| C — Contamination | decreasing | reduced by −0.20 penalty |
"""
EVAL_MD = """## Cell I: Evaluation Suite

Validates all 7 PRD evaluation dimensions:

| Dimension | Target |
|-----------|--------|
| Trend Classification Accuracy | ≥ 95% |
| Persistence Detection | 100% |
| Resistance Evolution Recall | 100% |
| Confidence Calibration (Brier) | ≤ 0.15 |
| Safety Compliance | 100% |
| Disclaimer Presence | 100% |
| Adversarial Robustness | 100% |
"""

GRADIO_MD = """## Cell J: Gradio UI — Extraction Agent

Interactive Gradio application with two entry modes:

- **Tab A — Upload PDF**: Upload one or more culture report PDFs. Docling parses each
  file into markdown, which is fed into the existing `extract_structured_data()` regex
  layer. Extracted records are shown in an editable review table before analysis.
- **Tab B — Enter Manually**: Paste free-text culture reports directly (existing flow).

The three-screen state machine (Upload → Review & Confirm → Analysis) is implemented
entirely via `gr.State` + `gr.update(visible=…)`. The downstream pipeline
(`analyze_trend`, `generate_hypothesis`, `call_medgemma`, `render_*`) is unchanged.
"""

FOOTER_MD = """---

## Safety & Regulatory Positioning

- **No output** from any module, in any mode, shall contain a named diagnosis.
- Confidence scores are capped at **0.95** (never 1.0 — clinical epistemic humility).
- Both output modes end with **hardcoded disclaimer text** that cannot be overridden.
- MedGemma is **never prompted with raw user text** — only structured JSON.
- A post-processing safety scan using `BANNED_DIAGNOSTIC_PHRASES` provides a second layer of defence.

> *This notebook is a Kaggle competition prototype only. It is not intended for clinical use,
> does not constitute medical advice, and has not been evaluated for diagnostic accuracy.*
"""

# ============================================================
# Assemble cells
# ============================================================

cells = [
    md_cell(TITLE_MD),
    md_cell(SETUP_MD),
    code_cell(SETUP_CODE),
    code_cell(IMPORTS_CODE),
    # B: Data Models (standalone — no local imports)
    md_cell(MODELS_MD),
    code_cell(inline("data_models.py")),
    # B cont: Rules (standalone — no local imports)
    code_cell(inline("rules.py")),
    # C: Extraction Layer
    md_cell(RULES_MD),
    code_cell(inline("extraction.py")),
    code_cell("# --- Extraction Unit Tests ---\n" + inline("test_extraction.py")),
    # D: Trend Engine
    md_cell(TREND_MD),
    code_cell(inline("trend.py")),
    code_cell("# --- Trend Unit Tests ---\n" + inline("test_trend.py")),
    # E: Hypothesis Layer
    md_cell(HYPOTHESIS_MD),
    code_cell(inline("hypothesis.py")),
    code_cell("# --- Hypothesis Unit Tests ---\n" + inline("test_hypothesis.py")),
    # F: MedGemma Integration
    md_cell(MEDGEMMA_MD),
    code_cell(inline("medgemma.py")),
    # G: Renderer
    md_cell(RENDERER_MD),
    code_cell(inline("renderer.py")),
    # H: Demo
    md_cell(DEMO_MD),
    code_cell(inline("demo.py")),
    # I: Evaluation Suite
    md_cell(EVAL_MD),
    code_cell(
        inline("evaluation.py")
        + "\n\n# Run evaluation\nreport = run_eval_suite()\nreport.print_report()"
    ),
    # J: Gradio UI — Extraction Agent
    md_cell(GRADIO_MD),
    code_cell(inline("extraction_agent.py")),
    code_cell(
        "# Launch the CultureSense Gradio app\n"
        "demo = build_gradio_app(model, tokenizer, is_stub)\n"
        "demo.launch(share=True)"
    ),
    md_cell(FOOTER_MD),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

out_path = os.path.join(BASE, "culturesense.ipynb")
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(cells)}")
