# CultureSense

A longitudinal clinical hypothesis engine for sequential urine and stool culture lab reports. Built as a Kaggle HAI-DEF competition submission using MedGemma (google/medgemma-4b-it) as the medical reasoning model, orchestrated by deterministic Python rule logic.

---

## Overview

CultureSense accepts 2-3 sequential culture lab reports and produces structured, non-diagnostic interpretations through two output modes:

- **Patient Mode** — plain-language trend summary with suggested questions for a doctor visit
- **Clinician Mode** — structured trajectory hypothesis with confidence score, resistance timeline, and stewardship alerts

The architecture is hybrid: deterministic rules handle all signal extraction and confidence scoring; MedGemma handles only natural language generation from pre-structured JSON inputs. Raw report text is never forwarded to the model.

---

## Architecture

```
[1] Raw Report Ingestion
    List[str] — 2-3 free-text culture reports
          |
          v
[2] Structured Extraction Layer
    extract_structured_data() -> CultureReport
          |
          v
[3] Temporal Comparison Engine
    analyze_trend() -> TrendResult
          |
          v
[4] Hypothesis Update Layer
    generate_hypothesis() -> HypothesisResult
    confidence [0.0 - 0.95]
          |
          v
[5] MedGemma Reasoning Layer
    call_medgemma(structured_payload, mode) -> str
    Modes: patient | clinician
          |
          v
[6] Structured Safe Output Renderer
    render_output() -> FormattedOutput
```

**Safety invariant:** Raw report text is never passed to MedGemma. Only derived, typed dataclass fields serialized to JSON are forwarded to the model.

---

## Repository Structure

```
culturesense/
├── CultureSense_PRD.md       # Product requirements document (master spec)
├── requirements.txt          # Python package dependencies
│
├── data_models.py            # Typed dataclasses for all data structures
├── rules.py                  # Clinical thresholds, organism alias table
├── extraction.py             # Free-text report parser
├── trend.py                  # Temporal CFU trend classifier
├── hypothesis.py             # Deterministic confidence scoring and risk flags
├── medgemma.py               # MedGemma integration with GPU/stub fallback
├── renderer.py               # Patient and Clinician output formatters
├── demo.py                   # 3-scenario demo runner
├── evaluation.py             # 7-dimension evaluation suite
│
├── test_extraction.py        # Extraction layer unit tests (20 tests)
├── test_trend.py             # Trend engine unit tests (15 tests)
├── test_hypothesis.py        # Hypothesis layer unit tests (17 tests)
│
├── build_notebook.py         # Assembles culturesense.ipynb from source files
└── culturesense.ipynb        # Kaggle-ready notebook (generated)
```

---

## Data Models

All data flows through five typed dataclasses defined in `data_models.py`:

| Dataclass | Purpose |
|-----------|---------|
| `CultureReport` | One parsed lab report — date, organism, CFU, resistance markers, specimen type, contamination flag, raw text |
| `TrendResult` | Temporal comparison across reports — CFU trend label, deltas, persistence, resistance evolution |
| `HypothesisResult` | Confidence score, risk flags, stewardship alert, interpretation string |
| `MedGemmaPayload` | Structured JSON payload for model inference (raw text explicitly excluded) |
| `FormattedOutput` | Final rendered output for either mode |

---

## Clinical Rule Engine

All scoring is deterministic and defined in `rules.py`. Confidence scoring starts at a base of 0.50 and applies fixed signal deltas:

| Signal | Delta |
|--------|-------|
| CFU decreasing | +0.30 |
| CFU cleared (final <= 1000 CFU/mL) | +0.40 |
| CFU increasing | +0.20 |
| CFU fluctuating | -0.10 |
| Resistance evolution detected | -0.10 |
| Organism changed between reports | -0.05 |
| Contamination present | -0.20 |
| Fewer than 2 reports | -0.25 |

Confidence is hard-capped at 0.95. A score of 1.0 is never produced.

CFU trend labels (in priority order): `cleared`, `decreasing`, `increasing`, `fluctuating`, `insufficient_data`.

---

## Safety Properties

- Raw report text is stored in `CultureReport.raw_text` and never included in `MedGemmaPayload`
- No output in any mode contains a named diagnosis
- Confidence is capped at 0.95 in all cases
- `requires_clinician_review` is always `True` — a structural field, not a runtime decision
- Both output modes append hardcoded disclaimer text that is not configurable at runtime
- A post-processing scan against `BANNED_DIAGNOSTIC_PHRASES` provides a second safety layer
- All four adversarial injection scenarios (prompt injection, SQL injection, diagnosis injection, organism-field injection) pass without leakage

---

## Evaluation Results

The evaluation suite (`evaluation.py`) covers 7 dimensions and produces 27 checks:

| Dimension | Target | Result |
|-----------|--------|--------|
| Trend Classification Accuracy | >= 95% | 6/6 PASS |
| Persistence Detection | 100% | 4/4 PASS |
| Resistance Evolution Recall | 100% | 4/4 PASS |
| Confidence Calibration (Brier Score) | <= 0.15 | 0.05 calibrated mean |
| Safety Compliance | 100% | 3/3 PASS |
| Disclaimer Presence | 100% | 2/2 PASS |
| Adversarial Robustness | 100% | 4/4 PASS |

**27/27 checks pass.**

---

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the evaluation suite:

```bash
python evaluation.py
```

Run the demo (3 scenarios, stub mode without GPU):

```bash
python demo.py
```

Run individual unit test suites:

```bash
python test_extraction.py
python test_trend.py
python test_hypothesis.py
```

Rebuild the Kaggle notebook from source:

```bash
python build_notebook.py
```

---

## Kaggle Deployment

The notebook `culturesense.ipynb` is designed for Kaggle with the following settings:

- **Accelerator:** GPU T4 x2
- **Internet:** Enabled (required for HuggingFace model download)
- **Python:** 3.11+

On CPU-only kernels (or if model loading fails for any reason), the stub fallback in `medgemma.py` activates automatically. All tests and the evaluation suite still run to completion in stub mode.

The notebook is self-contained — all source modules are inlined as sequential cells sharing one kernel namespace. No external files or imports from this repository are required at Kaggle runtime.

---

## Dependencies

```
transformers>=4.40.0
accelerate>=0.29.0
sentencepiece>=0.1.99
huggingface_hub>=0.22.0
torch>=2.2.0
```

Standard library only for the rule engine, extraction, trend, hypothesis, and renderer modules. `torch` and `transformers` are guarded by `try/except` throughout and are only required for live MedGemma inference.

---

## Disclaimer

This project is a Kaggle competition prototype. It is not intended for clinical use, does not constitute medical advice, and has not been evaluated for diagnostic accuracy. All outputs are non-diagnostic and require review by a qualified clinician.
