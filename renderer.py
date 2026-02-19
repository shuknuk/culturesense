"""
CultureSense Output Renderer (Cell Group G)

Produces FormattedOutput for both Patient and Clinician modes,
and provides display_output() for HTML-rendered Kaggle notebook display.
"""

from __future__ import annotations

from typing import Optional

from data_models import TrendResult, HypothesisResult, FormattedOutput

# ---------------------------------------------------------------------------
# G-1: Renderer Constants (Section 9.2–9.4, 9.6)
# ---------------------------------------------------------------------------

TREND_PHRASES: dict[str, str] = {
    "decreasing": "a downward trend in bacterial count",
    "cleared": "resolution of detectable bacteria",
    "increasing": "an upward trend in bacterial count",
    "fluctuating": "a variable pattern in bacterial count",
    "insufficient_data": "only one data point available",
}

PATIENT_QUESTIONS: list[str] = [
    "Is this trend consistent with my symptoms improving?",
    "Do I need another follow-up culture test?",
    "Are there any signs of antibiotic resistance I should know about?",
]

PATIENT_DISCLAIMER: str = (
    "IMPORTANT: This is an educational interpretation only. "
    "It is NOT a medical diagnosis. "
    "Please discuss all lab results with your healthcare provider."
)

CLINICIAN_DISCLAIMER: str = (
    "This output represents a structured hypothesis for clinical review. "
    "It is NOT a diagnosis and does NOT replace clinical judgment. "
    "All interpretations require full patient context and physician evaluation."
)


# ---------------------------------------------------------------------------
# G-2: render_patient_output()
# ---------------------------------------------------------------------------


def render_patient_output(
    trend: TrendResult,
    hypothesis: HypothesisResult,
    medgemma_response: str,
) -> FormattedOutput:
    """
    Construct a FormattedOutput for Patient Mode.

    Args:
        trend:             TrendResult from trend engine.
        hypothesis:        HypothesisResult from hypothesis layer.
        medgemma_response: String from call_medgemma() in 'patient' mode.

    Returns:
        FormattedOutput with patient_* fields populated.
        patient_disclaimer is ALWAYS appended unconditionally.
    """
    trend_phrase = TREND_PHRASES.get(trend.cfu_trend, "an uncertain pattern")
    confidence_note = f"Interpretation confidence: {hypothesis.confidence:.2f}"

    # Cap MedGemma explanation to ~150 words (soft limit)
    explanation_words = medgemma_response.split()
    if len(explanation_words) > 150:
        explanation = " ".join(explanation_words[:150]) + "..."
    else:
        explanation = medgemma_response

    return FormattedOutput(
        mode="patient",
        patient_trend_phrase=trend_phrase,
        patient_explanation=f"{explanation}\n\n{confidence_note}",
        patient_questions=list(PATIENT_QUESTIONS),
        patient_disclaimer=PATIENT_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# G-3: render_clinician_output()
# ---------------------------------------------------------------------------


def render_clinician_output(
    trend: TrendResult,
    hypothesis: HypothesisResult,
    medgemma_response: str,
) -> FormattedOutput:
    """
    Construct a FormattedOutput for Clinician Mode.

    Args:
        trend:             TrendResult from trend engine.
        hypothesis:        HypothesisResult from hypothesis layer.
        medgemma_response: String from call_medgemma() in 'clinician' mode.

    Returns:
        FormattedOutput with clinician_* fields populated.
        resistance_detail is only populated when resistance markers are present.
        clinician_disclaimer is ALWAYS appended unconditionally.
    """
    trajectory_summary: dict = {
        "report_dates": trend.report_dates,
        "cfu_values": trend.cfu_values,
        "cfu_deltas": trend.cfu_deltas,
        "cfu_trend": trend.cfu_trend,
        "organism_list": trend.organism_list,
        "organism_persistent": trend.organism_persistent,
        "any_contamination": trend.any_contamination,
        "resistance_evolution": trend.resistance_evolution,
    }

    # Build resistance detail only when resistance markers are present
    resistance_detail: Optional[str] = None
    has_any_resistance = any(markers for markers in trend.resistance_timeline)
    if has_any_resistance:
        lines = []
        for date, markers in zip(trend.report_dates, trend.resistance_timeline):
            marker_str = ", ".join(markers) if markers else "None"
            lines.append(f"  {date}: {marker_str}")
        resistance_detail = "Resistance Timeline:\n" + "\n".join(lines)

    return FormattedOutput(
        mode="clinician",
        clinician_trajectory=trajectory_summary,
        clinician_interpretation=medgemma_response,
        clinician_confidence=hypothesis.confidence,
        clinician_resistance_detail=resistance_detail,
        clinician_stewardship_flag=hypothesis.stewardship_alert,
        clinician_disclaimer=CLINICIAN_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# G-4: display_output()  — HTML-formatted Kaggle notebook rendering
# ---------------------------------------------------------------------------


def display_output(
    patient_out: FormattedOutput,
    clinician_out: FormattedOutput,
    scenario_name: str = "Culture Analysis",
) -> None:
    """
    Pretty-print both FormattedOutput objects using IPython HTML display.

    Falls back to plain-text print() when IPython is unavailable
    (e.g., running tests from the CLI).
    """
    html = _build_html(patient_out, clinician_out, scenario_name)

    try:
        from IPython.display import display, HTML

        display(HTML(html))
    except ImportError:
        # CLI / non-notebook fallback
        _print_plain(patient_out, clinician_out, scenario_name)


def _build_html(
    patient_out: FormattedOutput,
    clinician_out: FormattedOutput,
    scenario_name: str,
) -> str:
    """Build the HTML string for Kaggle notebook cell output."""

    # ---- Patient section ----
    questions_html = "".join(
        f"<li>{q}</li>" for q in (patient_out.patient_questions or [])
    )

    resistance_html = ""
    if clinician_out.clinician_resistance_detail:
        resistance_html = f"""
        <div style="background:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:8px 0;border-radius:4px;">
          <strong>Resistance Timeline</strong>
          <pre style="margin:4px 0;font-size:13px;">{clinician_out.clinician_resistance_detail}</pre>
        </div>
        """

    stewardship_html = ""
    if clinician_out.clinician_stewardship_flag:
        stewardship_html = """
        <div style="background:#f8d7da;border-left:4px solid #dc3545;padding:10px;margin:8px 0;border-radius:4px;">
          <strong>⚠ Stewardship Alert:</strong> Emerging resistance detected.
          Antimicrobial stewardship review recommended.
        </div>
        """

    traj = clinician_out.clinician_trajectory or {}
    traj_rows = "".join(
        f"<tr><td style='padding:4px 8px;border:1px solid #dee2e6;font-weight:bold;'>{k}</td>"
        f"<td style='padding:4px 8px;border:1px solid #dee2e6;'>{v}</td></tr>"
        for k, v in traj.items()
    )

    confidence_pct = (
        f"{clinician_out.clinician_confidence:.2f} "
        f"({clinician_out.clinician_confidence * 100:.0f}%)"
        if clinician_out.clinician_confidence is not None
        else "N/A"
    )

    html = f"""
    <div style="font-family:'Crimson Pro', serif;max-width:900px;margin:auto;color:#3d2b1f;">
      <h2 style="text-align:center;color:#d35400;border-bottom:2px solid #e67e22;padding-bottom:8px;font-variant:small-caps;">
        CultureSense — {scenario_name}
      </h2>

      <!-- PATIENT MODE -->
      <div style="background:#fffaf0;border-radius:12px;padding:20px;margin-bottom:20px;border:1px solid #fceec7;border-left:6px solid #f39c12;box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
        <h3 style="color:#d35400;margin-top:0;font-variant:small-caps;">Patient Mode</h3>
        <p style="font-size:1.1em;"><strong>Trend:</strong> Your results show <em>{patient_out.patient_trend_phrase}</em>.</p>
        <div style="line-height:1.6;color:#5d4037;">
          <strong>Summary:</strong><br>{(patient_out.patient_explanation or "").replace(chr(10), "<br>")}
        </div>
        <p style="margin-top:15px;"><strong>Questions to ask your doctor:</strong></p>
        <ul style="padding-left:20px;color:#5d4037;">{questions_html.replace('<li>', '<li style="margin-bottom:8px;">')}</ul>
        <div style="background:#fff3cd;padding:12px;border-radius:6px;border-left:4px solid #ffc107;font-style:italic;font-size:0.9em;margin-top:20px;">
          <strong>{patient_out.patient_disclaimer}</strong>
        </div>
      </div>

      <!-- CLINICIAN MODE -->
      <div style="background:#fefefe;border-radius:12px;padding:20px;border:1px solid #eee;border-left:6px solid #795548;box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
        <h3 style="color:#5d4037;margin-top:0;font-variant:small-caps;">Clinician Mode</h3>
        <p><strong>Confidence Score:</strong> <span style="font-size:1.2em;color:#e67e22;font-weight:bold;">{confidence_pct}</span></p>
        {stewardship_html}
        {resistance_html}
        <details style="margin-top:10px;border:1px solid #f5f5f5;border-radius:4px;padding:5px;">
          <summary style="cursor:pointer;font-weight:bold;color:#795548;">View Trajectory Data</summary>
          <table style="border-collapse:collapse;width:100%;margin-top:8px;font-size:13px;font-family:sans-serif;">
            {traj_rows}
          </table>
        </details>
        <p style="margin-top:15px;line-height:1.6;color:#3d2b1f;"><strong>Clinical Interpretation:</strong><br>
          {(clinician_out.clinician_interpretation or "").replace(chr(10), "<br>")}
        </p>
        <p style="font-style:italic;color:#8d6e63;border-top:1px solid #efebe9;padding-top:10px;margin-top:15px;font-size:0.85em;">
          {clinician_out.clinician_disclaimer}
        </p>
      </div>
    </div>
    """
    return html


def _print_plain(
    patient_out: FormattedOutput,
    clinician_out: FormattedOutput,
    scenario_name: str,
) -> None:
    """Plain-text fallback printer for non-notebook environments."""
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  CultureSense — {scenario_name}")
    print(sep)

    print("\n--- PATIENT MODE ---")
    print(f"Trend : {patient_out.patient_trend_phrase}")
    print(f"\n{patient_out.patient_explanation}")
    print("\nQuestions to ask your doctor:")
    for i, q in enumerate(patient_out.patient_questions or [], 1):
        print(f"  {i}. {q}")
    print(f"\n[!] {patient_out.patient_disclaimer}")

    print("\n--- CLINICIAN MODE ---")
    conf = clinician_out.clinician_confidence
    print(
        f"Confidence : {conf:.2f} ({conf * 100:.0f}%)"
        if conf is not None
        else "Confidence: N/A"
    )
    if clinician_out.clinician_stewardship_flag:
        print("[STEWARDSHIP ALERT] Emerging resistance — review recommended.")
    if clinician_out.clinician_resistance_detail:
        print(clinician_out.clinician_resistance_detail)
    if clinician_out.clinician_trajectory:
        print("Trajectory:")
        for k, v in clinician_out.clinician_trajectory.items():
            print(f"  {k}: {v}")
    print(f"\n{clinician_out.clinician_interpretation}")
    print(f"\n[i] {clinician_out.clinician_disclaimer}")
    print(sep)
