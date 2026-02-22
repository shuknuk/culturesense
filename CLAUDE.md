# CultureSense — Claude Code Project Memory

## Project Identity

**Team:** Kinshuk Goel (Lead) + Amit Goel
**GitHub:** https://github.com/shuknuk/culturesense
**Kaggle notebook:** culturesense.ipynb (auto-generated — never edit directly)

CultureSense is an **AI-powered clinical reasoning assistant** for urine and stool
culture reports. It processes 2–3 sequential reports and interprets how an infection
is changing over time through two distinct output modes:

- **Clinician Mode:** ranked hypotheses, resistance trend analysis, stewardship alerts
- **Patient Mode:** empathetic plain-language explanation, 5 structured discussion questions

It is NOT a diagnostic tool. It never diagnoses, never prescribes, never replaces
clinical judgment. The sole AI model is **MedGemma 4b-it** (google/medgemma-4b-it).

---

## Competition Context

**Competition:** Kaggle MedGemma Impact Challenge (HAI-DEF)
**URL:** https://www.kaggle.com/competitions/med-gemma-impact-challenge

**What judges score (priority order):**
1. Effective and creative use of MedGemma as a reasoning model — not summarization
2. Clinical grounding — real medical workflow understanding
3. Safety framing — non-diagnostic, explicit disclaimers, no autonomous decisions
4. Technical novelty — longitudinal reasoning across sequential reports
5. Feasibility — must run on Kaggle GPU T4 x2

**Prize track:** Agentic Workflow Prize — structured orchestration between
deterministic rule logic and MedGemma reasoning.

**Judge-facing positioning:**
> "CultureSense is not a chatbot. It is a longitudinal clinical hypothesis engine that
> interprets infection trajectories, detects resistance evolution, ranks hypotheses with
> confidence scores, and serves two distinct user personas — all while maintaining strict
> non-diagnostic safety boundaries. MedGemma only sees structured JSON, never raw text.
> This is the architectural safety guarantee and the technical novelty claim."

**2-minute video script (use verbatim for submission):**
> "Hello, we are Kinshuk Goel and Amit Goel. CultureSense is an AI-powered reasoning
> assistant designed specifically for urine and stool culture reports. Microbiology
> reports are complex. Clinicians must interpret organism identification, susceptibility
> tables, and evolving resistance patterns across multiple encounters. At the same time,
> patients often struggle to understand these reports, leading to confusion about
> antibiotic decisions. CultureSense addresses both challenges. In Clinician Mode, the
> system parses structured culture data, compares sequential reports, detects resistance
> evolution, and generates ranked clinical hypotheses with antimicrobial stewardship
> alerts. In Patient Mode, the same report is translated into simple, empathetic
> language. The system does not diagnose or prescribe. Instead, it explains what the
> findings mean and generates structured questions patients can discuss with their
> doctor. This reduces misunderstanding and supports informed conversations. Our hybrid
> approach uses prompt-engineered MedGemma combined with deterministic extraction and
> guardrails to ensure transparent, structured outputs. CultureSense enhances
> interpretability, antimicrobial stewardship awareness, and patient communication —
> while preserving clinical authority. Thank you."

---

## System Architecture (Final)

```
PDF Upload (Docling)  ──┐
Manual Text Input  ─────┴──► [1] PII Removal Layer
                                 - detect_pii() for audit logging
                                 - scrub_pii() to redact all PHI
                                 - [REDACTED] markers preserve structure
                                      |
                             [2] Structured Extraction
                                 - Primary: Regex patterns (fast, deterministic)
                                 - Fallback: MedGemma LLM extraction on regex failure
                                 - Organism, CFU/mL, date, specimen, resistance markers
                                 - Returns CultureReport dataclass
                                      |
                             [3] Sequential Comparison Engine
                                 - CFU trend
                                 - Organism persistence
                                 - Resistance evolution
                                      |
                             [4] MedGemma Reasoning Agent
                                 (structured JSON ONLY — never raw text)
                                      |
                             [5] Dual Output Layer
                                 |-- Clinician View
                                 └── Patient View
```

**Full function-level pipeline:**
```
[1] Raw Input → pii_removal.scrub_pii() → extract_structured_data() → CultureReport dataclass
    └─ Fallback: extract_structured_data_with_fallback() → MedGemma extraction on regex failure
[2] analyze_trend()            → TrendResult dataclass
[3] generate_hypothesis()      → HypothesisResult dataclass  (deterministic rules only)
[4] call_medgemma()            → str  (structured JSON payload — never raw text)
[5] render_clinician_output()  → FormattedOutput
    render_patient_output()    → FormattedOutput
```

**Notebook cell architecture:**
- Cell A: pip installs + imports
- Cell B: Data models (dataclasses) + RULES dict + ORGANISM_ALIASES
- Cell C: PII removal layer
- Cell D: Extraction layer (regex + extract_structured_data() + extract_structured_data_with_fallback())
- Cell E: Trend engine (analyze_trend())
- Cell F: Hypothesis layer (generate_hypothesis())
- Cell G: MedGemma integration (model loading + prompts + call_medgemma())
- Cell H: Output renderer (render_patient_output() + render_clinician_output())
- Cell I: Extraction agent (Docling PDF parsing + Review & Confirm screen)
- Cell J: Gradio UI (full dashboard)
- Cell K: Demo runner (3 test report sets)
- Cell L: Evaluation suite

---

## File Map

| File | Purpose |
|------|---------|
| `build_notebook.py` | Assembles source files into `culturesense.ipynb` — **always run after changes** |
| `culturesense.ipynb` | Kaggle notebook — **never edit directly, always auto-generated** |
| `data_models.py` | All dataclasses: CultureReport, TrendResult, HypothesisResult, MedGemmaPayload, FormattedOutput |
| `pii_removal.py` | PII removal — strips patient name, DOB, MRN before any processing |
| `extraction.py` | RULES dict, ORGANISM_ALIASES, regex patterns, extract_structured_data(), extract_structured_data_with_fallback() |
| `trend.py` | analyze_trend(), check_persistence(), check_resistance_evolution(), compute_deltas() |
| `hypothesis.py` | generate_hypothesis(), confidence scoring, risk flag assignment |
| `medgemma.py` | Model loading, system prompts, build_medgemma_payload(), call_medgemma() |
| `renderer.py` | TREND_PHRASES, disclaimers, render_patient_output(), render_clinician_output() |
| `extraction_agent.py` | Full Gradio dashboard + Docling PDF parsing — Upload + Manual tabs, Review & Confirm, Output |
| `heatmap.py` | Resistance timeline heatmap visualization (optional matplotlib) |
| `demo.py` | Local test runner — 3 hardcoded report sets |
| `evaluation.py` | Eval suite — trend accuracy, Brier score, safety compliance, adversarial tests |
| `test_extraction.py` | Unit tests — extraction layer |
| `test_trend.py` | Unit tests — trend engine |
| `test_hypothesis.py` | Unit tests — hypothesis layer |
| `CultureSense_PRD.md` | Full PRD — source of truth for all specs |
| `testPDFs/` | Synthetic Quest Diagnostics-style test PDFs |

---

## Absolute Architecture Rules

Never violate these regardless of what the task asks.

1. **MedGemma is the ONLY LLM.** No OpenAI, Anthropic, Gemini, or any other API calls.

2. **Raw text NEVER reaches MedGemma.** `call_medgemma()` receives only structured JSON
   built from typed dataclasses. `raw_text` on CultureReport is always `""` at call time.

3. **PII is removed before any processing.** Patient name, DOB, MRN are stripped by
   `pii_removal.py` before the text reaches `extract_structured_data()`.

4. **Confidence NEVER reaches 1.0.** Hard ceiling: `RULES["max_confidence"] = 0.95`.

5. **Disclaimers always last, never overrideable.** Both PATIENT_DISCLAIMER and
   CLINICIAN_DISCLAIMER are hardcoded constants appended unconditionally.

6. **No diagnosis language in any output.** BANNED_DIAGNOSTIC_PHRASES in evaluation.py
   is the enforcement list — treat as a hard constraint.

7. **Docling is a parser, not an LLM.** Produces markdown that feeds regex extraction.
   Never generates clinical content.

8. **extraction_agent.py only produces CultureReport objects.** Does not touch anything
   downstream. Pipeline from trend.py onward is a black box to it.

9. **MedGemma model:** `google/medgemma-4b-it` — never 7b on Kaggle T4.

10. **torch dtype:** `bfloat16` — never `float16`.

---

## Clinician Mode — Required Output Structure

MedGemma must return **ranked hypotheses** — not a single interpretation:

```
Hypothesis 1: [Name]
  Supporting Evidence:
    - [point 1]
    - [point 2]
  Confidence: 0.78

Hypothesis 2: [Name]
  Supporting Evidence:
    - [point 1]
  Confidence: 0.42
```

**All four sections required in Clinician output:**
1. Ranked hypotheses (minimum 2, each with confidence score)
2. Sequential trend panel (organism consistency, resistance shifts, escalation flags)
3. Resistance evolution heatmap (antibiotics Y-axis, time X-axis, S/I/R values)
4. Stewardship alert (if triggered — Clinician Mode only)

**Stewardship alert fires when ANY of these are true:**
- Resistance increases between sequential reports (S→I, S→R, or I→R)
- 2-class resistance detected (resistant to 2+ antibiotic classes)
- ESBL, CRE, MRSA, VRE, or CRKP markers confirmed
- Recurrent same organism within 30 days

---

## Patient Mode — Required Output Structure

All five sections are required every time:

**1. What This Report Shows**
Plain language, no organism names, no jargon.

**2. What This May Mean**
Contextualizes without diagnosing.

**3. Why Antibiotics May or May Not Be Used**
Two branches — if prescribed and if not prescribed.

**4. Questions to Discuss With Your Doctor (exactly 5):**
1. Is this bacteria definitely causing my symptoms?
2. Why was this specific antibiotic chosen?
3. Do I need a repeat culture later?
4. What symptoms should prompt urgent evaluation?
5. Is this likely to happen again?

**5. Reassurance Statement:**
"This explanation is intended to help you understand your report. Your doctor has full
knowledge of your medical history and is best placed to guide your care."

---

## Confidence Scoring Algorithm

Starting confidence: `0.50`

| Signal | Delta |
|--------|-------|
| CFU trend = "decreasing" | +0.30 |
| CFU trend = "cleared" (≤ 1000 CFU/mL) | +0.40 |
| CFU trend = "increasing" | +0.20 |
| CFU trend = "fluctuating" | −0.10 |
| Resistance evolution detected | −0.10 |
| Organism changed between reports | −0.05 |
| Contamination flag present | −0.20 |
| Only 1 report (insufficient data) | −0.25 |

Final confidence clamped to `[0.0, 0.95]`. Never 1.0.

---

## MedGemma — Model & Generation Spec

```python
MODEL_NAME = "google/medgemma-4b-it"   # NEVER 7b
torch_dtype = torch.bfloat16            # NEVER float16
device_map = "auto"

max_new_tokens = 512
temperature = 0.3
top_p = 0.9
do_sample = True
repetition_penalty = 1.1
```

---

## Known Persistent Bugs — Read Before Touching These Files

These two bugs have survived multiple fix attempts. Root causes documented below.
Do not patch around them — replace the broken logic entirely with what is written here.

### BUG A — Organism persistence returning False (trend.py)

**Symptom:** ORGANISM_CHANGE flag fires even when all reports show Escherichia coli.

**Root cause:** check_persistence() runs set() on raw strings before normalization.
Case differences and alias variants cause false negatives.

**Replace check_persistence() with exactly this — do not modify it:**

```python
def check_persistence(organism_list: List[str]) -> bool:
    normalized = [
        ORGANISM_ALIASES.get(o.strip().lower(), o.strip().lower())
        for o in organism_list
    ]
    return len(set(normalized)) == 1
```

**Also verify ORGANISM_ALIASES has these exact entries (all lowercase keys):**

```python
ORGANISM_ALIASES = {
    "e. coli":           "escherichia coli",
    "e.coli":            "escherichia coli",
    "escherichia coli":  "escherichia coli",  # identity — required
    "klebsiella":        "klebsiella pneumoniae",
    "staph aureus":      "staphylococcus aureus",
    "enterococcus":      "enterococcus faecalis",
    "mixed flora":       "mixed flora",
    "mixed growth":      "mixed flora",
    "skin flora":        "mixed flora",
    "normal flora":      "mixed flora",
}
```

The identity mapping `"escherichia coli": "escherichia coli"` is required.
Without it, .get() falls back to the raw string and case mismatches break equality.

### BUG B — Resistance timeline showing single letters (FIXED)

**Status:** ✅ FIXED — defensive type checking added to `render_resistance_timeline()`

**Symptom:** Timeline renders "C m O n R s S i" — these are initials from antibiotic
susceptibility interpretation strings, not resistance markers.

**Root cause:** Renderer is iterating over susceptibility dicts instead of
TrendResult.resistance_timeline. Additionally, data serialization through Gradio State
can convert nested lists to strings, causing iteration over individual characters.

**Fix applied in `extraction_agent.py`:**

```python
def render_resistance_timeline(trend: TrendResult) -> str:
    # Defensive: ensure resistance_timeline is List[List[str]]
    timeline = trend.resistance_timeline
    report_dates = trend.report_dates

    # Handle case where data might be serialized/deserialized through Gradio State
    if isinstance(timeline, str):
        import json
        try:
            timeline = json.loads(timeline)
        except (json.JSONDecodeError, TypeError):
            timeline = []

    if isinstance(report_dates, str):
        import json
        try:
            report_dates = json.loads(report_dates)
        except (json.JSONDecodeError, TypeError):
            report_dates = []

    # Ensure timeline is a list
    if not isinstance(timeline, list):
        timeline = []

    # Ensure report_dates is a list
    if not isinstance(report_dates, list):
        report_dates = []

    has_any = any(
        len(markers) > 0 if isinstance(markers, (list, tuple)) else bool(markers)
        for markers in timeline
    )

    if not has_any:
        return "No high-risk resistance markers detected."

    rows = []
    for date, markers in zip(report_dates, timeline):
        # Handle case where markers might be a string instead of list
        if isinstance(markers, str):
            markers = [markers] if markers else []
        marker_str = ", ".join(markers) if markers else "None"
        rows.append(f"| {date} | {marker_str} |")

    header = ("| Date | High-Risk Resistance Markers |\n"
              "|------|------------------------------|")
    return header + "\n" + "\n".join(rows)
```

**Key defensive measures:**
1. Type-check `resistance_timeline` and `report_dates` before iteration
2. JSON-deserialize strings that may have been converted through Gradio State
3. Handle case where markers might be a string instead of list
4. Always default to empty lists on parse failures

### BUG C — Stewardship Alert firing on cleared trend (FIXED)

**Status:** ✅ FIXED — modified stewardship logic in `hypothesis.py` line 171

**Symptom:** Stewardship Alert shows "⚠ Emerging resistance detected" even when
infection has cleared (trend = "cleared"). Patient sees both "✓ Resolution Detected"
and "⚠ Stewardship Alert" which is confusing.

**Root cause:** `stewardship_alert = trend.resistance_evolution` fired whenever
resistance evolved, regardless of whether the infection had since cleared.

**Fix applied in `hypothesis.py`:**

```python
# Stewardship alert: only fire if resistance evolved AND infection hasn't cleared
# If trend is cleared, historical resistance is not an active stewardship concern
stewardship_alert = trend.resistance_evolution and trend.cfu_trend not in ("cleared",)
```

### BUG D — Contradictory interpretation for cleared trend (FIXED)

**Status:** ✅ FIXED — modified interpretation logic in `hypothesis.py` lines 139-142

**Symptom:** Patient explanation contains contradictory statements:
- "organisms is now zero in all samples" (cleared)
- "we've also seen some changes in the type of organism present" (organism change)

**Root cause:** `_build_interpretation()` added organism change message even when
trend was cleared. If the infection has cleared, organism persistence is irrelevant.

**Fix applied in `hypothesis.py`:**

```python
# Only mention organism change if trend is not cleared
# (if cleared, organism persistence is irrelevant - the infection has resolved)
if not trend.organism_persistent and trend.cfu_trend != "cleared":
    parts.append("Organism change may indicate reinfection.")
```

---

## PII/PHI Removal — Pattern Reference

The `pii_removal.py` module provides defense-in-depth PII protection.

### PII Patterns Scrubbed

| Type | Pattern Example | Replacement |
|------|-----------------|-------------|
| Patient Name | `Patient Name: John Smith` | `Patient Name: [REDACTED NAME]` |
| DOB | `DOB: 01/15/1980` | `DOB: [REDACTED DOB]` |
| MRN | `MRN: 12345678` | `MRN: [REDACTED MRN]` |
| SSN | `SSN: 123-45-6789` | `[REDACTED SSN]` |
| Phone | `Phone: (555) 123-4567` | `[REDACTED PHONE]` |
| Email | `patient@email.com` | `[REDACTED EMAIL]` |
| Address | `Address: 123 Main St` | `Address: [REDACTED ADDRESS]` |
| Provider | `Provider: Dr. Smith` | `Provider: [REDACTED PROVIDER]` |

### API Usage

```python
from pii_removal import scrub_pii, detect_pii

# Detect what PII is present (for audit logging)
detected = detect_pii(report_text)
# Returns: ["name", "dob", "mrn"]

# Scrub PII from text
clean_text = scrub_pii(report_text)
# Returns: text with all PII replaced by [REDACTED ...] markers

# Also scrub provider names
clean_text = scrub_pii(report_text, remove_provider_names=True)
```

### Key Design Decisions

1. **Preserves document structure**: [REDACTED] tokens maintain line structure for regex extraction
2. **Regex-based only**: No LLM involvement in PII detection (deterministic, fast, safe)
3. **Logged but not stored**: `detect_pii()` returns types found, not actual values
4. **Runs before extraction**: All text reaching extraction is guaranteed PII-free

---

## MedGemma Fallback Extraction

When regex extraction fails, `extraction.py` provides an optional MedGemma fallback.

### Function Signature

```python
def extract_structured_data_with_fallback(
    report_text: str,
    medgemma_model=None,
    medgemma_tokenizer=None,
    use_medgemma_fallback: bool = True
) -> CultureReport:
```

### Extraction Flow

```
report_text
    |
    ▼
┌─────────────────────────┐
│ Regex extraction        │
│ extract_structured_data │
└─────────────────────────┘
    │ Success ──► return CultureReport
    │ Failure
    ▼
┌─────────────────────────┐     ┌─────────────────┐
│ MedGemma fallback       │────►│ Build prompt    │
│ (if model available)    │     │ asking for JSON │
└─────────────────────────┘     │ with 5 fields   │
    │                           └─────────────────┘
    │                                       │
    ▼                                       ▼
┌─────────────────────────┐     ┌─────────────────┐
│ Parse JSON response     │◄────│ MedGemma        │
│ into CultureReport      │     │ generate()      │
└─────────────────────────┘     └─────────────────┘
```

### Prompt Structure

MedGemma is prompted to extract exactly these fields:
```json
{
  "organism": "E. coli",
  "cfu": 100000,
  "date": "2024-01-15",
  "specimen_type": "urine",
  "resistance_markers": ["ESBL"]
}
```

### Safety Considerations

- MedGemma fallback only processes **PII-scrubbed** text (defense in depth)
- Fallback result has `raw_text=""` (never stores the text used for LLM extraction)
- Temperature 0.1 for deterministic extraction
- Max 256 new tokens to prevent over-generation

---

## Safety Compliance — Banned Phrases

```python
BANNED_DIAGNOSTIC_PHRASES = [
    "you have", "you are diagnosed", "the diagnosis is",
    "confirms infection", "you should take", "prescribe",
    "definitive diagnosis", "this is a urinary tract infection"
]
```

---

## Test PDFs and Expected Outputs (testPDFs/)

| File | Scenario | Expected output |
|------|----------|----------------|
| `QuestDx_Report1_Week1_ActiveUTI.pdf` | E. coli 150k CFU, cipro prescribed | baseline, confidence≈0.50 |
| `QuestDx_Report2_Week2_MidTreatment.pdf` | 45k CFU, cipro MIC→Intermediate | stewardship_alert=True |
| `QuestDx_Report3_Week3_Resolution.pdf` | 3k CFU, MIC restored | trend=cleared, confidence≥0.75, NO flags |
| `SetA_Improving_EColi.pdf` | 120k→40k→5k, no resistance | trend=decreasing, confidence≥0.80 |
| `SetB_EmergingResistance_ESBL.pdf` | ESBL confirmed report 3 | resistance_evolution=True, stewardship=True |
| `SetC_Contamination_Report1/2.pdf` | Mixed flora, 2 separate PDFs | contamination_flag=True, confidence≈0.25 |
| `SetD_MixedLabReport_FilterTest.pdf` | CBC + CMP + urine mixed | only urine extracted |

**Validation checklist for the 3 QuestDx reports:**
- [ ] Step 2: single table render (not duplicated), 3 rows
- [ ] Step 2: dates 2026-02-01 / 2026-02-08 / 2026-02-15 (not "unknown")
- [ ] Step 2: all resistance markers show —
- [ ] Step 3 patient: no resistance mentioned, resolution messaging, all 5 sections present
- [ ] Step 3 clinician: confidence ≥ 0.75
- [ ] Step 3 clinician: NO ORGANISM_CHANGE flag
- [ ] Step 3 clinician: NO EMERGING_RESISTANCE flag
- [ ] Step 3 clinician: NO stewardship alert
- [ ] Step 3 clinician: resistance timeline shows "No high-risk resistance markers detected"

---

## Workflow Rules

- **After any code changes**, always run:
  ```bash
  python3 build_notebook.py
  ```

## Testing

```bash
# Run all unit tests
python3 test_pii_removal.py && python3 test_extraction.py && python3 test_trend.py && python3 test_hypothesis.py

# Run evaluation suite
python3 evaluation.py
```

All tests must pass before build_notebook.py is run.

## Code Conventions

- Regex patterns live in `extraction.py` constants section only
- `bfloat16` not `float16`
- `google/medgemma-4b-it` not `7b`
- Document any new regex patterns with inline comments

## Common Tasks

- Debug PDF extraction: `debug_extraction()` in `extraction.py`
- Test locally: `python3 demo.py`
- Verify PDF parsing: upload to Gradio Upload tab — dates must not show as "unknown"
