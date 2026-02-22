"""
CultureSense Demo Runner (Cell Group H)
Runs 3 pre-built simulated report scenarios and renders output.

Scenario A: Improving infection (decreasing → cleared)
Scenario B: Emerging resistance (Klebsiella + ESBL)
Scenario C: Contamination (mixed flora, low CFU)
"""

from data_models import CultureReport
from trend import analyze_trend
from hypothesis import generate_hypothesis
from medgemma import load_medgemma, call_medgemma
from renderer import render_patient_output, render_clinician_output, display_output

# ---------------------------------------------------------------------------
# Load MedGemma once (stub fallback if no GPU)
# ---------------------------------------------------------------------------
print("Loading MedGemma model ...")
model, tokenizer, is_stub = load_medgemma()
if is_stub:
    print("Running with stub fallback (no GPU detected or model unavailable).")
else:
    print("MedGemma loaded on GPU.")


def run_scenario(
    name: str,
    reports: list[CultureReport],
    expected_notes: str = "",
) -> None:
    """
    Full pipeline: trend → hypothesis → MedGemma → render → display.
    """
    print(f"\n{'=' * 60}")
    print(f"Scenario: {name}")
    if expected_notes:
        print(f"Expected: {expected_notes}")
    print("=" * 60)

    # Sort by date (oldest first)
    sorted_reports = sorted(reports, key=lambda r: r.date)

    # Pipeline
    trend = analyze_trend(sorted_reports)
    hypothesis = generate_hypothesis(trend, len(sorted_reports))

    patient_response = call_medgemma(
        trend, hypothesis, "patient", model, tokenizer, is_stub, sorted_reports
    )
    clinician_response = call_medgemma(
        trend, hypothesis, "clinician", model, tokenizer, is_stub, sorted_reports
    )

    patient_out = render_patient_output(trend, hypothesis, patient_response, sorted_reports)
    clinician_out = render_clinician_output(trend, hypothesis, clinician_response, sorted_reports)

    display_output(patient_out, clinician_out, scenario_name=name)

    # Print structured diagnostics
    print(
        f"\n[Diagnostics]  trend={trend.cfu_trend}  "
        f"confidence={hypothesis.confidence:.2f}  "
        f"flags={hypothesis.risk_flags}  "
        f"stewardship={hypothesis.stewardship_alert}"
    )


# ---------------------------------------------------------------------------
# Cell H-1: Scenario A — Improving Infection
# ---------------------------------------------------------------------------
scenario_a = [
    CultureReport(
        "2026-01-01", "Escherichia coli", 120000, [], [], "urine", False, "<raw>"
    ),
    CultureReport("2026-01-10", "Escherichia coli", 40000, [], [], "urine", False, "<raw>"),
    CultureReport("2026-01-20", "Escherichia coli", 5000, [], [], "urine", False, "<raw>"),
]

run_scenario(
    name="Scenario A — Improving Infection",
    reports=scenario_a,
    expected_notes="trend=decreasing, confidence≥0.80, Patient Mode reassuring, Clinician Mode clean trajectory",
)

# ---------------------------------------------------------------------------
# Cell H-2: Scenario B — Emerging Resistance
# ---------------------------------------------------------------------------
scenario_b = [
    CultureReport(
        "2026-01-01", "Klebsiella pneumoniae", 90000, [], [], "urine", False, "<raw>"
    ),
    CultureReport(
        "2026-01-10", "Klebsiella pneumoniae", 80000, [], [], "urine", False, "<raw>"
    ),
    CultureReport(
        "2026-01-20", "Klebsiella pneumoniae", 75000, ["ESBL"], [], "urine", False, "<raw>"
    ),
]

run_scenario(
    name="Scenario B — Emerging Resistance",
    reports=scenario_b,
    expected_notes="trend=fluctuating, resistance_evolution=True, stewardship_flag=True, confidence reduced",
)

# ---------------------------------------------------------------------------
# Cell H-3: Scenario C — Contamination
# ---------------------------------------------------------------------------
scenario_c = [
    CultureReport("2026-01-01", "mixed flora", 5000, [], [], "urine", True, "<raw>"),
    CultureReport("2026-01-10", "mixed flora", 3000, [], [], "urine", True, "<raw>"),
]

run_scenario(
    name="Scenario C — Contamination",
    reports=scenario_c,
    expected_notes="contamination in both, confidence~0.20, Patient Mode gentle, Clinician Mode flags contamination",
)

print("\n\nDemo run complete.")
