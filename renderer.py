"""
CultureSense Output Renderer (Cell Group G)

Produces FormattedOutput for both Patient and Clinician modes,
and provides display_output() for HTML-rendered Kaggle notebook display.
"""

from __future__ import annotations

from typing import Optional

from data_models import TrendResult, HypothesisResult, FormattedOutput

# Heatmap is optional - gracefully handle if matplotlib not available
try:
    from heatmap import generate_resistance_heatmap, get_heatmap_html
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False
    generate_resistance_heatmap = None
    get_heatmap_html = None

# Import heatmap module (optional - gracefully handles missing matplotlib)
try:
    from heatmap import generate_resistance_heatmap, get_heatmap_html
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False

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
        "multi_drug_resistance": trend.multi_drug_resistance,
    }

    # Build resistance detail only when resistance markers are present
    resistance_detail: Optional[str] = None

    # Defensive: Handle case where data might be serialized through Gradio State
    # Gradio may convert lists to Python literal strings (single quotes) not JSON
    report_dates = trend.report_dates
    resistance_timeline = trend.resistance_timeline

    if isinstance(report_dates, str):
        import json
        import ast
        try:
            report_dates = json.loads(report_dates)
        except (json.JSONDecodeError, TypeError):
            try:
                report_dates = ast.literal_eval(report_dates)
            except (ValueError, SyntaxError):
                report_dates = []

    if isinstance(resistance_timeline, str):
        import json
        import ast
        try:
            resistance_timeline = json.loads(resistance_timeline)
        except (json.JSONDecodeError, TypeError):
            try:
                resistance_timeline = ast.literal_eval(resistance_timeline)
            except (ValueError, SyntaxError):
                resistance_timeline = []

    # Ensure they are lists
    if not isinstance(report_dates, list):
        report_dates = []
    if not isinstance(resistance_timeline, list):
        resistance_timeline = []

    has_any_resistance = any(markers for markers in resistance_timeline)
    if has_any_resistance:
        lines = []
        for date, markers in zip(report_dates, resistance_timeline):
            # Handle case where markers might be a string instead of list
            if isinstance(markers, str):
                markers = [markers] if markers else []
            marker_str = ", ".join(markers) if markers else "None"
            lines.append(f"  {date}: {marker_str}")
        resistance_detail = "Resistance Timeline:\n" + "\n".join(lines)

    # Generate resistance heatmap if matplotlib is available
    resistance_heatmap: Optional[str] = None
    if has_any_resistance and generate_resistance_heatmap is not None:
        resistance_heatmap = generate_resistance_heatmap(
            trend.resistance_timeline,
            trend.report_dates
        )

    return FormattedOutput(
        mode="clinician",
        clinician_trajectory=trajectory_summary,
        clinician_interpretation=medgemma_response,
        clinician_confidence=hypothesis.confidence,
        clinician_resistance_detail=resistance_detail,
        clinician_resistance_heatmap=resistance_heatmap,
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

    # ---- Resistance / stewardship ----
    resistance_html = ""
    if clinician_out.clinician_resistance_detail:
        resistance_html = f"""
        <div style="background:#F5F0EB;border-left:3px solid #E8DDD6;padding:10px 14px;margin:10px 0;border-radius:3px;">
          <p style="margin:0 0 4px 0;font-family:system-ui,sans-serif;font-size:0.8rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:#7A6558;">Resistance Timeline</p>
          <pre style="margin:0;font-size:12px;font-family:system-ui,monospace;color:#4A3728;white-space:pre-wrap;">{clinician_out.clinician_resistance_detail}</pre>
        </div>
        """

    stewardship_html = ""
    if clinician_out.clinician_stewardship_flag:
        stewardship_html = """
        <div style="background:#FDF5F1;border-left:3px solid #C1622F;padding:10px 14px;margin:10px 0;border-radius:3px;">
          <span style="font-family:system-ui,sans-serif;font-size:0.85rem;color:#C1622F;font-weight:600;">Stewardship Alert</span>
          <p style="margin:4px 0 0 0;font-family:system-ui,sans-serif;font-size:0.82rem;color:#5D4037;">Emerging resistance detected — antimicrobial stewardship review recommended.</p>
        </div>
        """

    # ---- Trajectory table ----
    traj = clinician_out.clinician_trajectory or {}
    traj_rows = "".join(
        f"<tr>"
        f"<td style='padding:5px 10px;border-bottom:1px solid #E8DDD6;border-right:1px solid #E8DDD6;"
        f"font-family:'Source Serif 4',serif;font-size:0.78rem;font-weight:600;color:#7A6558;"
        f"text-transform:uppercase;letter-spacing:.03em;white-space:nowrap;'>{k}</td>"
        f"<td style='padding:5px 10px;border-bottom:1px solid #E8DDD6;"
        f"font-family:'Source Serif 4',serif;font-size:0.82rem;color:#4A3728;'>{v}</td>"
        f"</tr>"
        for k, v in traj.items()
    )

    # ---- Confidence bar ----
    conf_val = clinician_out.clinician_confidence
    conf_pct_num = int((conf_val or 0) * 100)
    conf_label = (
        f"{conf_val:.0%}" if conf_val is not None else "N/A"
    )
    conf_bar_html = f"""
    <div style="margin:12px 0 16px;">
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:5px;">
        <span style="font-family:system-ui,sans-serif;font-size:0.78rem;font-weight:600;color:#7a6558;text-transform:uppercase;letter-spacing:.04em;">Confidence</span>
        <span style="font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;color:#C1622F;">{conf_label}</span>
      </div>
      <div style="height:5px;border-radius:3px;background:#E8DDD6;overflow:hidden;">
        <div style="height:100%;width:{conf_pct_num}%;background:#C1622F;border-radius:3px;"></div>
      </div>
    </div>
    """

    # ---- Google Fonts import ----
    font_import = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&'
        'family=Source+Serif+4:ital,wght@0,400;0,500;1,400;1,500&display=swap" rel="stylesheet">'
    )

    html = f"""
    {font_import}
    <div style="font-family:'Source Serif 4',serif;max-width:860px;margin:auto;color:#4A3728;background:#FDFAF7;padding:28px 32px;border:1px solid #E8DDD6;border-radius:4px;">

      <!-- Page header -->
      <div style="text-align:center;border-bottom:1px solid #E8DDD6;padding-bottom:16px;margin-bottom:24px;">
        <h2 style="font-family:'Playfair Display',serif;font-weight:700;font-size:1.55rem;color:#C1622F;margin:0 0 4px 0;letter-spacing:.01em;">
          CultureSense
        </h2>
        <p style="font-family:system-ui,sans-serif;font-size:0.8rem;color:#7A6558;margin:0;letter-spacing:.06em;text-transform:uppercase;">{scenario_name}</p>
      </div>

      <!-- PATIENT MODE -->
      <section style="margin-bottom:28px;padding-bottom:24px;border-bottom:1px solid #E8DDD6;">
        <h3 style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:600;color:#C1622F;margin:0 0 14px 0;letter-spacing:.01em;border-left:3px solid #C1622F;padding-left:10px;">Patient Summary</h3>
        <p style="font-size:1.0rem;line-height:1.75;margin:0 0 12px 0;"><em>Your results show <strong>{patient_out.patient_trend_phrase}</strong>.</em></p>
        <div style="line-height:1.75;color:#4A3728;font-size:0.97rem;">
          {(patient_out.patient_explanation or "").replace(chr(10), "<br>")}
        </div>
        <p style="margin:16px 0 6px 0;font-family:system-ui,sans-serif;font-size:0.78rem;font-weight:600;color:#7A6558;text-transform:uppercase;letter-spacing:.05em;">Questions to ask your doctor</p>
        <ul style="padding-left:18px;color:#4A3728;font-size:0.94rem;line-height:1.85;margin:0;">
          {questions_html.replace('<li>', '<li style="margin-bottom:4px;">')}
        </ul>
        <div style="margin-top:18px;padding:10px 14px;border:1px solid #E8DDD6;border-radius:3px;background:#F5F0EB;">
          <p style="font-family:system-ui,sans-serif;font-size:0.78rem;font-style:italic;color:#9A8578;margin:0;line-height:1.6;">{patient_out.patient_disclaimer}</p>
        </div>
      </section>

      <!-- CLINICIAN MODE -->
      <section>
        <h3 style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:600;color:#C1622F;margin:0 0 14px 0;letter-spacing:.01em;border-left:3px solid #C1622F;padding-left:10px;">Clinical Interpretation</h3>
        {conf_bar_html}
        {stewardship_html}
        {resistance_html}
        <details style="margin:12px 0;border:1px solid #E8DDD6;border-radius:3px;">
          <summary style="cursor:pointer;padding:8px 12px;font-family:system-ui,sans-serif;font-size:0.8rem;font-weight:600;color:#7A6558;text-transform:uppercase;letter-spacing:.04em;list-style:none;user-select:none;">Trajectory Data</summary>
          <div style="padding:0 12px 12px;">
            <table style="border-collapse:collapse;width:100%;margin-top:8px;border:1px solid #E8DDD6;">
              {traj_rows}
            </table>
          </div>
        </details>
        <div style="line-height:1.75;color:#4A3728;font-size:0.97rem;margin-top:14px;">
          {(clinician_out.clinician_interpretation or "").replace(chr(10), "<br>")}
        </div>
        <p style="font-family:system-ui,sans-serif;font-style:italic;color:#7A6558;border-top:1px solid #E8DDD6;padding-top:12px;margin-top:20px;font-size:0.77rem;line-height:1.6;">
          {clinician_out.clinician_disclaimer}
        </p>
      </section>

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
