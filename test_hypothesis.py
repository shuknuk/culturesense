"""
CultureSense Hypothesis Layer — Unit Tests (Cell E-2)
Kaggle-native inline tests (no pytest dependency).
"""

from data_models import CultureReport, TrendResult
from trend import analyze_trend
from hypothesis import generate_hypothesis, FLAG_EMERGING_RESISTANCE, FLAG_CONTAMINATION

_PASS = 0
_FAIL = 0


def _assert(condition: bool, msg: str) -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {msg}")
    else:
        _FAIL += 1
        print(f"  FAIL  {msg}")


def _make_report(
    cfu: int,
    organism: str = "Escherichia coli",
    date: str = "2026-01-01",
    markers=None,
    contamination: bool = False,
) -> CultureReport:
    return CultureReport(
        date=date,
        organism=organism,
        cfu=cfu,
        resistance_markers=markers or [],
        specimen_type="urine",
        contamination_flag=contamination,
        raw_text="<stub>",
    )


# ---------------------------------------------------------------------------
# 1. Perfect improvement (decreasing → cleared) — confidence ≥ 0.80
# ---------------------------------------------------------------------------
print("=== Test: Perfect Improvement (Decreasing → Cleared) ===")
rpts = [
    _make_report(120000, date="2026-01-01"),
    _make_report(40000, date="2026-01-10"),
    _make_report(800, date="2026-01-20"),  # cleared (≤ 1000)
]
trend = analyze_trend(rpts)
hyp = generate_hypothesis(trend, len(rpts))

_assert(
    hyp.confidence >= 0.80,
    f"confidence ≥ 0.80 for cleared trend  (got {hyp.confidence})",
)
_assert(
    hyp.confidence <= 0.95, f"confidence ≤ 0.95 (hard ceiling)  (got {hyp.confidence})"
)
_assert(hyp.stewardship_alert is False, f"stewardship_alert == False")
_assert(hyp.requires_clinician_review is True, f"requires_clinician_review always True")
_assert(
    "possible resolution" in hyp.interpretation, f"interpretation mentions resolution"
)

# ---------------------------------------------------------------------------
# 2. Emerging resistance — confidence drops vs. clean improving scenario
# ---------------------------------------------------------------------------
print("\n=== Test: Emerging Resistance (Confidence Drops) ===")
rpts2 = [
    _make_report(90000, date="2026-01-01", markers=[]),
    _make_report(80000, date="2026-01-10", markers=[]),
    _make_report(75000, date="2026-01-20", markers=["ESBL"]),
]
trend2 = analyze_trend(rpts2)
hyp2 = generate_hypothesis(trend2, len(rpts2))

_assert(
    FLAG_EMERGING_RESISTANCE in hyp2.risk_flags, f"EMERGING_RESISTANCE in risk_flags"
)
_assert(hyp2.stewardship_alert is True, f"stewardship_alert == True")
_assert(
    hyp2.confidence < 0.80,
    f"confidence < 0.80 when resistance emerges  (got {hyp2.confidence})",
)

# ---------------------------------------------------------------------------
# 3. Contamination — confidence is reduced by the -0.20 contamination penalty.
#    With decreasing CFU (5000→3000): base 0.50 + 0.30 (decreasing) - 0.20 (contamination) = 0.60
#    The PRD Appendix B example uses a fluctuating pattern; here decreasing gives 0.60.
# ---------------------------------------------------------------------------
print("\n=== Test: Contamination (Confidence Drops Sharply) ===")
rpts3 = [
    _make_report(5000, organism="mixed flora", date="2026-01-01", contamination=True),
    _make_report(3000, organism="mixed flora", date="2026-01-10", contamination=True),
]
trend3 = analyze_trend(rpts3)
hyp3 = generate_hypothesis(trend3, len(rpts3))

_assert(FLAG_CONTAMINATION in hyp3.risk_flags, f"CONTAMINATION_SUSPECTED in risk_flags")
_assert(
    hyp3.confidence <= 0.65,
    f"confidence reduced by contamination penalty (got {hyp3.confidence})",
)
_assert(
    "Contamination suspected" in hyp3.interpretation,
    f"interpretation flags contamination",
)

# ---------------------------------------------------------------------------
# 4. Single report — insufficient data penalty
# ---------------------------------------------------------------------------
print("\n=== Test: Single Report (Insufficient Data) ===")
rpts4 = [_make_report(100000, date="2026-01-01")]
trend4 = analyze_trend(rpts4)
hyp4 = generate_hypothesis(trend4, len(rpts4))

_assert(
    hyp4.confidence == 0.25,
    f"confidence == 0.25 (base 0.50 - 0.25)  (got {hyp4.confidence})",
)
_assert("INSUFFICIENT_DATA" in hyp4.risk_flags, f"INSUFFICIENT_DATA in risk_flags")

# ---------------------------------------------------------------------------
# 5. Increasing CFU — non-response pattern
# ---------------------------------------------------------------------------
print("\n=== Test: Increasing CFU (Non-Response) ===")
rpts5 = [
    _make_report(40000, date="2026-01-01"),
    _make_report(80000, date="2026-01-10"),
    _make_report(120000, date="2026-01-20"),
]
trend5 = analyze_trend(rpts5)
hyp5 = generate_hypothesis(trend5, len(rpts5))

_assert(
    "NON_RESPONSE_PATTERN" in hyp5.risk_flags, f"NON_RESPONSE_PATTERN in risk_flags"
)
_assert(
    hyp5.confidence == 0.70,
    f"confidence == 0.70 (0.50 + 0.20)  (got {hyp5.confidence})",
)
_assert(
    "non-response" in hyp5.interpretation.lower(),
    f"interpretation mentions non-response",
)

# ---------------------------------------------------------------------------
# 6. Confidence never exceeds 0.95
# ---------------------------------------------------------------------------
print("\n=== Test: Confidence Hard Ceiling ===")
# Best possible scenario: cleared, persistent, no resistance, no contamination
rpts6 = [
    _make_report(120000, date="2026-01-01"),
    _make_report(800, date="2026-01-10"),  # cleared
]
trend6 = analyze_trend(rpts6)
hyp6 = generate_hypothesis(trend6, len(rpts6))
_assert(
    hyp6.confidence <= 0.95, f"confidence never exceeds 0.95  (got {hyp6.confidence})"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 50}")
print(f"Hypothesis Tests Complete: {_PASS} passed, {_FAIL} failed")
if _FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {_FAIL} test(s) failed")
