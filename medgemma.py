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
""".strip()

# ---------------------------------------------------------------------------
# Payload builder (Section 8.5)
# raw_text is NEVER included — only derived structured fields
# ---------------------------------------------------------------------------


def build_medgemma_payload(
    trend: TrendResult,
    hypothesis: HypothesisResult,
    mode: str,
) -> str:
    """
    Build a JSON string to pass as the user turn to MedGemma.

    IMPORTANT: raw_text from CultureReport is explicitly excluded.
    Only deterministic derived fields are forwarded.

    Args:
        trend:      Computed TrendResult.
        hypothesis: Computed HypothesisResult.
        mode:       "patient" | "clinician"

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

        flags_note = ""
        if trend.resistance_evolution:
            flags_note = (
                " Your doctor may want to discuss the latest results in detail."
            )

        return (
            f"Your lab results show {trend_desc} over the time period reviewed. "
            f"This information has been summarised for your awareness.{flags_note} "
            "Please discuss these findings with your doctor."
        )

    else:  # clinician
        flags = ", ".join(hypothesis.risk_flags) if hypothesis.risk_flags else "None"
        stewardship = (
            "ALERT: Antimicrobial stewardship review recommended."
            if hypothesis.stewardship_alert
            else ""
        )
        return (
            f"Trajectory Hypothesis Summary\n"
            f"CFU Trend: {trend.cfu_trend}\n"
            f"Organism Persistent: {trend.organism_persistent}\n"
            f"Resistance Evolution: {trend.resistance_evolution}\n"
            f"Confidence: {hypothesis.confidence:.2f} ({hypothesis.confidence * 100:.0f}%)\n"
            f"Risk Flags: {flags}\n"
            f"{stewardship}\n"
            f"Interpretation: {hypothesis.interpretation}\n"
            "Clinical interpretation requires full patient context."
        ).strip()


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
    user_content = build_medgemma_payload(trend, hypothesis, mode)

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
