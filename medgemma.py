"""
CultureSense MedGemma Integration (Cell Group F)

Handles:
  - Model loading with GPU fallback stub
  - System prompt constants for Patient and Clinician modes
  - Structured payload construction (raw_text NEVER included)
  - call_medgemma() inference with generation parameters from Section 8.6
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from typing import Optional

from data_models import TrendResult, HypothesisResult

# ---------------------------------------------------------------------------
# Model ID
# ---------------------------------------------------------------------------
MODEL_ID = "google/medgemma-4b-it"  # Instruction-tuned variant

# ---------------------------------------------------------------------------
# System prompts (Section 8.3 / 8.4)
# ---------------------------------------------------------------------------

PATIENT_SYSTEM_PROMPT = """
You are a compassionate medical communication assistant.
You are given STRUCTURED DATA only --- not raw patient reports.
Your task: Generate a plain-language explanation of a lab result trend.

STRICT RULES:
1. NEVER diagnose. Never say "you have X".
2. NEVER recommend a treatment or medication.
3. Always end with: "Please discuss these findings with your doctor."
4. Use empathetic, reassuring language.
5. Respond ONLY based on the structured data provided.
6. Do not reference specific bacteria names to the patient.
7. When describing CFU values, use ONLY the exact numbers from cfu_values. Do not round, approximate, or change the values in any way.
8. If resistance_timeline shows no markers, explicitly state there are no signs of antibiotic resistance.
9. When susceptibility_profiles is provided, analyze which antibiotics are SENSITIVE (will work) vs RESISTANT (will not work). Explain this in plain language: "The bacteria responded to X antibiotics" or "The bacteria did not respond to Y antibiotics." Do not use medical abbreviations like S/I/R.
10. Never mention specific antibiotic names (e.g., Ciprofloxacin, Nitrofurantoin, Ampicillin, Ceftriaxone, etc.). Do not list drug names. Instead say "some antibiotics were tested" or "your doctor has the full antibiotic results".
""".strip()

CLINICIAN_SYSTEM_PROMPT = """
You are a structured clinical decision support assistant.
You are given STRUCTURED TEMPORAL DATA from a rule-based analysis engine.
Your task: Generate a structured trajectory interpretation for a clinician.

STRICT RULES:
1. Frame all outputs as hypotheses, not diagnoses.
2. Always include confidence score in output.
3. Flag stewardship concerns explicitly if resistance_evolution is True.
4. End with: "Clinical interpretation requires full patient context."
5. Use clinical terminology appropriate for a physician audience.
6. Never recommend a specific antibiotic or treatment regimen.
7. When susceptibility_profiles is provided, analyze antimicrobial susceptibility patterns. Identify which antibiotic classes are effective (Sensitive) vs ineffective (Resistant). Note any multi-drug resistance patterns. Include MIC values where clinically relevant.
8. You MUST return exactly 2 ranked hypotheses. Never return a single paragraph. Format:

Hypothesis 1: [name]
  Supporting Evidence:
    - [Cite specific data from reports: e.g., "Report 1 (2024-01-15): E. coli at 150,000 CFU/mL"]
    - [Include actual values: e.g., "CFU trajectory: 150,000 → 45,000 → 5,000 (decreasing)"]
  Confidence: [0.0-0.95]

Hypothesis 2: [name]
  Supporting Evidence:
    - [Cite specific data from reports with dates and values]
  Confidence: [0.0-0.95]

CRITICAL: Every evidence point MUST cite actual values from the structured data:
- Reference report dates (e.g., "Report 1 (2024-01-15)")
- Include CFU counts with units (e.g., "150,000 CFU/mL")
- Note organism names (e.g., "Escherichia coli")
- Specify antibiotic sensitivities when available (e.g., "Sensitive to Ciprofloxacin")
- Never use generic phrases like "trend suggests" without citing the specific data points
""".strip()

# ---------------------------------------------------------------------------
# Payload builder (Section 8.5)
# raw_text is NEVER included — only derived structured fields
# ---------------------------------------------------------------------------


def _format_susceptibility_for_payload(reports: list) -> list[dict]:
    """
    Format susceptibility profiles from reports for MedGemma payload.

    Returns a list of report summaries with antibiotic susceptibility data.
    """
    result = []
    for report in reports:
        if hasattr(report, 'susceptibility_profile') and report.susceptibility_profile:
            antibiotics = []
            for s in report.susceptibility_profile:
                antibiotics.append({
                    "antibiotic": s.antibiotic,
                    "mic": s.mic,
                    "interpretation": s.interpretation
                })
            result.append({
                "date": report.date,
                "organism": report.organism,
                "cfu": report.cfu,
                "antibiotics": antibiotics
            })
    return result


def build_medgemma_payload(
    trend: TrendResult,
    hypothesis: HypothesisResult,
    mode: str,
    reports: list = None,
) -> str:
    """
    Build a JSON string to pass as the user turn to MedGemma.

    IMPORTANT: raw_text from CultureReport is explicitly excluded.
    Only deterministic derived fields are forwarded.

    Args:
        trend:      Computed TrendResult.
        hypothesis: Computed HypothesisResult.
        mode:       "patient" | "clinician"
        reports:    Optional list of CultureReport objects for susceptibility data.

    Returns:
        JSON string ready to embed in a chat message.
    """
    if mode not in ("patient", "clinician"):
        raise ValueError(f"mode must be 'patient' or 'clinician', got '{mode}'")

    payload = {
        "mode": mode,
        "cfu_trend": trend.cfu_trend,
        "cfu_values": trend.cfu_values,
        "cfu_deltas": trend.cfu_deltas,
        "organism_persistent": trend.organism_persistent,
        "resistance_evolution": trend.resistance_evolution,
        "resistance_timeline": trend.resistance_timeline,
        "any_contamination": trend.any_contamination,
        "report_dates": trend.report_dates,
        "interpretation": hypothesis.interpretation,
        "confidence": hypothesis.confidence,
        "risk_flags": hypothesis.risk_flags,
        "stewardship_alert": hypothesis.stewardship_alert,
        "requires_clinician_review": hypothesis.requires_clinician_review,
        # raw_text intentionally omitted — safety guarantee
    }

    # Include susceptibility data if reports provided
    if reports:
        payload["susceptibility_profiles"] = _format_susceptibility_for_payload(reports)

    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Model loading — with CPU fallback stub
# ---------------------------------------------------------------------------


def load_medgemma(
    model_id: str = MODEL_ID,
) -> tuple:
    """
    Attempt to load MedGemma from HuggingFace.

    Returns:
        (model, tokenizer, is_stub) tuple.
        is_stub=True means the stub fallback is active (no GPU / model unavailable).

    GPU note (Kaggle): accelerator=GPU T4 x2, bfloat16 reduces VRAM to ~4 GB.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        gpu_available = torch.cuda.is_available()
        if not gpu_available:
            warnings.warn(
                "No CUDA GPU detected. Activating MedGemma stub fallback. "
                "Outputs will be templated, not LLM-generated.",
                UserWarning,
                stacklevel=2,
            )
            return None, None, True

        print(f"Loading {model_id} on GPU ({torch.cuda.get_device_name(0)}) ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        print("MedGemma loaded successfully.")
        return model, tokenizer, False

    except Exception as exc:
        warnings.warn(
            f"MedGemma model loading failed ({exc}). Activating stub fallback.",
            UserWarning,
            stacklevel=2,
        )
        return None, None, True


# ---------------------------------------------------------------------------
# Stub fallback response templates
# ---------------------------------------------------------------------------


def _stub_response(mode: str, trend: TrendResult, hypothesis: HypothesisResult) -> str:
    """
    Return a hardcoded template response when MedGemma is unavailable.
    Used for CPU-only Kaggle kernels or when model loading fails.
    """
    if mode == "patient":
        trend_desc = {
            "decreasing": "a downward trend in your lab values",
            "cleared": "that your lab values have returned to a normal range",
            "increasing": "an upward trend in your lab values",
            "fluctuating": "a variable pattern in your lab values",
            "insufficient_data": "limited data — only one result is available",
        }.get(trend.cfu_trend, "an uncertain pattern in your lab values")

        # Build explanation without mentioning specific antibiotic names
        explanation_parts = []

        if trend.resistance_evolution:
            explanation_parts.append(
                "Some changes in antibiotic response were detected. Your doctor may want to discuss the latest results in detail."
            )
        elif trend.cfu_trend == "cleared":
            explanation_parts.append(
                "The bacterial count has dropped to very low levels. This may indicate that treatment has been effective."
            )
        elif trend.cfu_trend == "decreasing":
            explanation_parts.append(
                "The bacterial count is going down, which suggests the current approach is working."
            )
        elif trend.cfu_trend == "increasing":
            explanation_parts.append(
                "The bacterial count is rising. Your doctor may consider additional testing to identify the best approach."
            )
        else:
            explanation_parts.append(
                "Your doctor has the full test results and will discuss what this means for your care."
            )

        flags_note = " ".join(explanation_parts)

        return (
            f"Your lab results show {trend_desc} over the time period reviewed. "
            f"{flags_note} "
            "Please discuss these findings with your doctor."
        )

    else:  # clinician
        flags = ", ".join(hypothesis.risk_flags) if hypothesis.risk_flags else "None"
        stewardship = (
            "\nStewardship Alert: Antimicrobial stewardship review recommended."
            if hypothesis.stewardship_alert
            else ""
        )

        # Build evidence points from trend data
        evidence_points = []
        if trend.cfu_trend == "decreasing":
            evidence_points.append("CFU trend shows decreasing bacterial load")
        elif trend.cfu_trend == "cleared":
            evidence_points.append("CFU values have normalized")
        elif trend.cfu_trend == "increasing":
            evidence_points.append("CFU trend shows increasing bacterial load")

        if trend.organism_persistent:
            evidence_points.append("Organism persistence across reports")
        else:
            evidence_points.append("Organism variation between reports")

        if trend.resistance_evolution:
            evidence_points.append("Resistance markers detected")

        # Build first hypothesis (primary)
        primary_evidence = [f"  - {point}" for point in evidence_points[:2]]
        primary_evidence_str = "\n".join(primary_evidence) if primary_evidence else "  - Trend data available"

        # Build second hypothesis (alternative)
        alt_evidence = []
        if trend.cfu_trend == "insufficient_data":
            alt_evidence.append("  - Single report limits trend analysis")
        else:
            alt_evidence.append("  - Multiple reports provide trend context")

        if trend.any_contamination:
            alt_evidence.append("  - Contamination flag present")

        alt_evidence_str = "\n".join(alt_evidence) if alt_evidence else "  - Follow-up testing recommended"

        return (
            f"Hypothesis 1: {hypothesis.interpretation}\n"
            f"  Supporting Evidence:\n"
            f"{primary_evidence_str}\n"
            f"  Confidence: {hypothesis.confidence:.2f}\n\n"
            f"Hypothesis 2: Alternative Interpretation\n"
            f"  Supporting Evidence:\n"
            f"{alt_evidence_str}\n"
            f"  Confidence: {max(0.0, hypothesis.confidence - 0.25):.2f}\n"
            f"{stewardship}\n\n"
            "Risk Flags: " + flags + "\n"
            "Clinical interpretation requires full patient context."
        )


# ---------------------------------------------------------------------------
# Main inference function (Section F-4)
# ---------------------------------------------------------------------------


def call_medgemma(
    trend: TrendResult,
    hypothesis: HypothesisResult,
    mode: str,
    model=None,
    tokenizer=None,
    is_stub: bool = True,
    reports: list = None,
) -> str:
    """
    Call MedGemma with a fully structured JSON payload.

    If is_stub=True (no GPU / model unavailable), returns a templated
    fallback response so the notebook continues to execute end-to-end.

    Generation parameters (Section 8.6):
        max_new_tokens=512, temperature=0.3, top_p=0.9,
        do_sample=True, repetition_penalty=1.1

    Args:
        trend:      TrendResult from trend engine.
        hypothesis: HypothesisResult from hypothesis layer.
        mode:       "patient" | "clinician"
        model:      Loaded HuggingFace model (None if stub).
        tokenizer:  Loaded HuggingFace tokenizer (None if stub).
        is_stub:    True → use stub fallback.

    Returns:
        Decoded string response (special tokens stripped).
    """
    if is_stub or model is None or tokenizer is None:
        return _stub_response(mode, trend, hypothesis)

    import torch

    system_prompt = (
        PATIENT_SYSTEM_PROMPT if mode == "patient" else CLINICIAN_SYSTEM_PROMPT
    )
    user_content = build_medgemma_payload(trend, hypothesis, mode, reports)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Apply chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()
