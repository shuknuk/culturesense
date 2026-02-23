"""
CultureSense Hypothesis Update Layer (Cell Group E)
Deterministic confidence scoring and risk flag assignment.
"""

from typing import List

from data_models import TrendResult, HypothesisResult
from rules import RULES


# ---------------------------------------------------------------------------
# Risk flag constants
# ---------------------------------------------------------------------------
FLAG_EMERGING_RESISTANCE = "EMERGING_RESISTANCE"
FLAG_CONTAMINATION = "CONTAMINATION_SUSPECTED"
FLAG_NON_RESPONSE = "NON_RESPONSE_PATTERN"
FLAG_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
FLAG_ORGANISM_CHANGE = "ORGANISM_CHANGE"
FLAG_MULTI_DRUG_RESISTANCE = "MULTI_DRUG_RESISTANCE"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def _score_confidence(trend: TrendResult, report_count: int, has_symptom_data: bool = False) -> float:
    """
    Apply deterministic signal adjustments to base confidence value.

    New algorithm (Section 7.1, updated):
        - Start at 0.90 if organism, threshold, and susceptibility are clear
        - Subtract 0.20 if no longitudinal data (single report)
        - Subtract 0.20 if no symptom data
        - Clamp to [min_confidence, max_confidence] = [0.20, 0.95]

    Legacy trend signals (still applied for longitudinal data):
        +0.30  CFU decreasing
        +0.40  CFU cleared
        +0.20  CFU increasing  (high confidence of non-response)
        -0.10  CFU fluctuating
        -0.10  resistance evolution
        -0.05  organism changed
        -0.20  contamination present
    """
    # Start with high base confidence (clear organism, threshold, susceptibility)
    confidence = RULES["confidence_high_base"]

    # Penalty: no longitudinal data (single report)
    if report_count < 2:
        confidence -= RULES["confidence_longitudinal_penalty"]

    # Penalty: no symptom data
    if not has_symptom_data:
        confidence -= RULES["confidence_symptom_penalty"]

    # Legacy trend signals (only apply if we have longitudinal data)
    if report_count >= 2:
        if trend.cfu_trend == "decreasing":
            confidence += 0.30
        elif trend.cfu_trend == "cleared":
            confidence += 0.40
        elif trend.cfu_trend == "increasing":
            confidence += 0.20  # high confidence of non-response
        elif trend.cfu_trend == "fluctuating":
            confidence -= 0.10

        # Resistance evolution penalty (only for longitudinal)
        if trend.resistance_evolution:
            confidence -= 0.10

        # Organism change uncertainty (only for longitudinal)
        if not trend.organism_persistent:
            confidence -= 0.05

    # Contamination validity concern (always applies)
    if trend.any_contamination:
        confidence -= 0.20

    # Hard clamp: never < min_confidence, never > max_confidence (epistemic humility)
    min_conf = RULES.get("min_confidence", 0.20)
    max_conf = RULES["max_confidence"]
    return round(max(min_conf, min(confidence, max_conf)), 4)


# ---------------------------------------------------------------------------
# Risk flag assignment (Section 7.2)
# ---------------------------------------------------------------------------


def _assign_risk_flags(trend: TrendResult, report_count: int) -> List[str]:
    """Build a list of risk flag strings from trend signals."""
    flags: List[str] = []

    if trend.resistance_evolution:
        flags.append(FLAG_EMERGING_RESISTANCE)

    if trend.any_contamination:
        flags.append(FLAG_CONTAMINATION)

    if trend.cfu_trend == "increasing":
        flags.append(FLAG_NON_RESPONSE)

    if report_count < 2:
        flags.append(FLAG_INSUFFICIENT_DATA)

    if not trend.organism_persistent:
        flags.append(FLAG_ORGANISM_CHANGE)

    if trend.multi_drug_resistance:
        flags.append(FLAG_MULTI_DRUG_RESISTANCE)

    return flags


# ---------------------------------------------------------------------------
# Interpretation string construction (Section 7.3)
# ---------------------------------------------------------------------------


def _build_interpretation(trend: TrendResult, report_count: int) -> str:
    """
    Construct a rule-generated natural language pattern summary.

    This string is passed to MedGemma only as structured context inside
    the JSON payload — never as a direct LLM prompt.
    """
    parts: List[str] = []

    if trend.cfu_trend == "decreasing":
        parts.append("Pattern suggests improving infection response.")
    elif trend.cfu_trend == "cleared":
        parts.append("Pattern suggests possible resolution.")
    elif trend.cfu_trend == "increasing":
        parts.append("Pattern suggests possible non-response.")
    elif trend.cfu_trend == "fluctuating":
        parts.append("Pattern is variable — requires clinical context.")
    elif trend.cfu_trend == "insufficient_data":
        parts.append("Insufficient longitudinal data for trend analysis.")

    if trend.resistance_evolution:
        parts.append("Emerging resistance observed.")

    # Only mention organism change if trend is not cleared
    # (if cleared, organism persistence is irrelevant - the infection has resolved)
    if not trend.organism_persistent and trend.cfu_trend != "cleared":
        parts.append("Organism change may indicate reinfection.")

    if trend.any_contamination:
        parts.append("Contamination suspected — interpret with caution.")

    if trend.multi_drug_resistance:
        parts.append("Multi-drug resistance pattern detected.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_hypothesis(trend: TrendResult, report_count: int) -> HypothesisResult:
    """
    Generate a deterministic hypothesis from a TrendResult.

    Args:
        trend: Computed TrendResult from the trend engine.
        report_count: Number of source reports (used for insufficient-data logic).

    Returns:
        HypothesisResult with confidence score, risk flags, interpretation,
        stewardship alert, and mandatory clinician review flag.
    """
    confidence = _score_confidence(trend, report_count)
    risk_flags = _assign_risk_flags(trend, report_count)
    interpretation = _build_interpretation(trend, report_count)
    # Stewardship alert fires when:
    # 1. Resistance EVOLUTION detected (new resistances appearing), OR
    # 2. Multi-drug resistance AND infection NOT improving (CFU not decreasing/cleared), OR
    # 3. Recurrent organism within 30 days
    # Note: Baseline MDR with improving infection does NOT trigger alert (treatment is working)
    stewardship_alert = (
        trend.cfu_trend not in ("cleared",)
        and (
            trend.resistance_evolution
            or (trend.multi_drug_resistance and trend.cfu_trend not in ("decreasing", "cleared"))
            or trend.recurrent_organism_30d
        )
    )

    return HypothesisResult(
        interpretation=interpretation,
        confidence=confidence,
        risk_flags=risk_flags,
        stewardship_alert=stewardship_alert,
        requires_clinician_review=True,  # Always True — structural safety guarantee
    )
