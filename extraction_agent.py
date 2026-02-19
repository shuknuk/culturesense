"""
CultureSense Extraction Agent (Cell J)
PDF upload flow: Docling → extract_structured_data() → Gradio UI

Three-screen state machine:
  Screen 1 — Upload PDFs
  Screen 2 — Review & Confirm extracted records (editable table)
  Screen 3 — Analysis output (existing pipeline, zero changes)

Tab B (manual entry) is the existing flow — zero modifications.
"""

import warnings
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Optional

import gradio as gr

from data_models import CultureReport
from extraction import extract_structured_data, ExtractionError
from trend import analyze_trend
from hypothesis import generate_hypothesis
from medgemma import call_medgemma
from renderer import render_patient_output, render_clinician_output
from rules import RULES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RECORDS = 3
_WARN_PREFIX = "⚠ "

# ---------------------------------------------------------------------------
# Theme Definition — "Orange Design Theme, Warm Classical UI"
# ---------------------------------------------------------------------------

WARM_CLINICAL_THEME = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Source Serif 4"), "serif"],
    font_mono=[gr.themes.GoogleFont("Source Code Pro"), "monospace"],
).set(
    body_background_fill="#FDFAF7",  # Warm white
    block_background_fill="#FDFAF7",
    block_border_width="1px",
    block_border_color="#E8DDD6",
    block_title_text_font="'Playfair Display', serif",
    button_primary_background_fill="#C1622F",
    button_primary_background_fill_hover="#a85228",
    button_primary_text_color="#FDFAF7",
)


# ---------------------------------------------------------------------------
# 1. Docling PDF processor
# ---------------------------------------------------------------------------

def process_pdf_file(pdf_path: str) -> Tuple[str, str]:
    """
    Parse a single PDF with Docling.

    Returns:
        (markdown_text, status_html)
        - On success: (markdown, "")  — caller checks for culture data
        - On parse failure: ("", "<red status>")
    """
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()
        return markdown_text, ""
    except Exception as e:
        return (
            "",
            f'<span style="color:#c0392b">✗ Could not read this file — '
            f'try a clearer scan or copy ({type(e).__name__})</span>',
        )


# ---------------------------------------------------------------------------
# 2. Multi-file orchestrator
# ---------------------------------------------------------------------------

def process_uploaded_pdfs(
    files: List,
) -> Tuple[List[CultureReport], List[str], List[str], str]:
    """
    Process a list of uploaded PDF file objects from gr.File.

    Returns:
        (reports, raw_text_blocks, per_file_statuses, truncation_warning)
        - reports: deduplicated, sorted, max MAX_RECORDS CultureReport list
        - raw_text_blocks: one markdown string per report (for clinician accordion)
        - per_file_statuses: one HTML status string per uploaded file
        - truncation_warning: non-empty string if records were truncated
    """
    if not files:
        return [], [], [], ""

    all_reports: List[CultureReport] = []
    all_raw_blocks: List[str] = []
    per_file_statuses: List[str] = []

    for f in files:
        # Gradio passes file objects with a .name attribute (temp path)
        pdf_path = f.name if hasattr(f, "name") else str(f)
        filename = Path(pdf_path).name

        markdown_text, parse_error = process_pdf_file(pdf_path)

        if parse_error:
            per_file_statuses.append(
                f'<div style="margin:4px 0"><b>{filename}</b> — {parse_error}</div>'
            )
            continue

        # Try to extract culture records from the markdown
        # extract_structured_data() handles one report block at a time.
        # For multi-report PDFs, split on common section delimiters.
        report_blocks = _split_into_report_blocks(markdown_text)
        file_reports: List[CultureReport] = []

        for block in report_blocks:
            try:
                report = extract_structured_data(block)
                # Only keep urine/stool specimens
                if report.specimen_type in ("urine", "stool"):
                    # Override raw_text to the docling markdown block
                    report = CultureReport(
                        date=report.date,
                        organism=report.organism,
                        cfu=report.cfu,
                        resistance_markers=report.resistance_markers,
                        specimen_type=report.specimen_type,
                        contamination_flag=report.contamination_flag,
                        raw_text=block,  # stored for accordion; never forwarded to MedGemma
                    )
                    file_reports.append(report)
            except ExtractionError:
                pass  # block had no parseable culture data

        if not file_reports:
            per_file_statuses.append(
                f'<div style="margin:4px 0"><b>{filename}</b> — '
                f'<span style="color:#e67e22">⚠ No urine or stool culture data found in this file</span></div>'
            )
        else:
            count = len(file_reports)
            per_file_statuses.append(
                f'<div style="margin:4px 0"><b>{filename}</b> — '
                f'<span style="color:#27ae60">✓ {count} record{"s" if count != 1 else ""} found</span></div>'
            )
            all_reports.extend(file_reports)
            all_raw_blocks.extend(r.raw_text for r in file_reports)

    if not all_reports:
        return [], [], per_file_statuses, ""

    # Sort chronologically
    combined = sorted(
        zip(all_reports, all_raw_blocks), key=lambda pair: pair[0].date
    )
    all_reports = [p[0] for p in combined]
    all_raw_blocks = [p[1] for p in combined]

    # Deduplicate: same (date, organism, cfu) → keep first
    seen: set = set()
    deduped_reports: List[CultureReport] = []
    deduped_blocks: List[str] = []
    for report, block in zip(all_reports, all_raw_blocks):
        key = (report.date, report.organism, report.cfu)
        if key in seen:
            warnings.warn(
                f"Duplicate record skipped: {key}", UserWarning, stacklevel=2
            )
        else:
            seen.add(key)
            deduped_reports.append(report)
            deduped_blocks.append(block)

    # Truncate to MAX_RECORDS most recent
    truncation_warning = ""
    if len(deduped_reports) > MAX_RECORDS:
        total = len(deduped_reports)
        deduped_reports = deduped_reports[-MAX_RECORDS:]
        deduped_blocks = deduped_blocks[-MAX_RECORDS:]
        truncation_warning = (
            f'<div style="background:#fff3cd;border:1px solid #ffc107;padding:8px 12px;'
            f'border-radius:6px;margin-bottom:8px">'
            f'⚠ {total} records were extracted. Only the {MAX_RECORDS} most recent are shown '
            f'(the pipeline supports up to {MAX_RECORDS} reports).</div>'
        )

    return deduped_reports, deduped_blocks, per_file_statuses, truncation_warning


def _split_into_report_blocks(markdown_text: str) -> List[str]:
    """
    Attempt to split a multi-report markdown document into individual report blocks.

    Heuristic: split on markdown H1/H2 headings or horizontal rules that
    typically separate reports. Falls back to returning the whole text as one block.
    """
    import re

    # Try splitting on "---" or "===" separators (common in lab report PDFs)
    blocks = re.split(r"\n(?:---+|===+)\n", markdown_text)
    if len(blocks) > 1:
        return [b.strip() for b in blocks if b.strip()]

    # Try splitting on H1/H2 headings
    blocks = re.split(r"\n(?=#{1,2} )", markdown_text)
    if len(blocks) > 1:
        return [b.strip() for b in blocks if b.strip()]

    # Single block
    return [markdown_text.strip()] if markdown_text.strip() else []


# ---------------------------------------------------------------------------
# 3. Confidence heuristic
# ---------------------------------------------------------------------------

def _is_low_confidence(report: CultureReport) -> bool:
    """Return True if the extracted record has low confidence."""
    return report.organism == "unknown" or (
        report.cfu == 0 and report.organism != "unknown"
    )


# ---------------------------------------------------------------------------
# 4. Dataframe bridge
# ---------------------------------------------------------------------------

def reports_to_dataframe_rows(reports: List[CultureReport]) -> List[List]:
    """Convert CultureReport list → list of rows for gr.Dataframe."""
    rows = []
    for r in reports:
        resistance_str = ", ".join(r.resistance_markers) if r.resistance_markers else "None"
        if _is_low_confidence(r):
            # Prefix organism and resistance with warning indicator
            organism_display = f"{_WARN_PREFIX}{r.organism}"
        else:
            organism_display = r.organism
        rows.append([
            r.date,
            r.specimen_type,
            organism_display,
            str(r.cfu),
            resistance_str,
        ])
    return rows


def dataframe_row_to_culture_report(row) -> CultureReport:
    """
    Convert an edited gr.Dataframe row back to a CultureReport.

    CRITICAL: raw_text is always set to "" — never forwarded to MedGemma.
    """
    date = str(row[0]).strip()
    specimen = str(row[1]).strip().lower()
    organism = str(row[2]).strip().replace(_WARN_PREFIX, "")
    cfu_raw = str(row[3]).replace(",", "").strip()
    resistance_raw = str(row[4]).strip()

    try:
        cfu = int(cfu_raw)
    except ValueError:
        cfu = 0

    resistance = [
        m.strip().replace(_WARN_PREFIX, "")
        for m in resistance_raw.split(",")
        if m.strip() and m.strip().lower() != "none"
    ]

    contamination_flag = any(
        term in organism.lower() for term in RULES["contamination_terms"]
    )

    return CultureReport(
        date=date,
        organism=organism,
        cfu=cfu,
        resistance_markers=resistance,
        specimen_type=specimen,
        contamination_flag=contamination_flag,
        raw_text="",  # SAFETY: never forwarded to MedGemma
    )


# ---------------------------------------------------------------------------
# 5. Gradio UI builder
# ---------------------------------------------------------------------------

def build_gradio_app(model, tokenizer, is_stub: bool) -> gr.Blocks:
    """
    Build and return the full CultureSense Gradio Blocks app.

    Tab A — Upload PDF (new extraction agent flow)
    Tab B — Enter Manually (existing flow, zero changes)
    """

    # ── Shared pipeline helper ──────────────────────────────────────────────
    def run_pipeline(reports: List[CultureReport]):
        """Run the unchanged downstream pipeline and return rendered HTML."""
        sorted_reports = sorted(reports, key=lambda r: r.date)
        trend = analyze_trend(sorted_reports)
        hypothesis = generate_hypothesis(trend, len(sorted_reports))
        patient_response = call_medgemma(trend, hypothesis, "patient", model, tokenizer, is_stub)
        clinician_response = call_medgemma(trend, hypothesis, "clinician", model, tokenizer, is_stub)
        patient_out = render_patient_output(trend, hypothesis, patient_response)
        clinician_out = render_clinician_output(trend, hypothesis, clinician_response)
        return patient_out, clinician_out

    def format_output_html(patient_out, clinician_out) -> Tuple[str, str]:
        """Convert FormattedOutput objects to display HTML — warm classical theme."""
        # ── Patient card ───────────────────────────────────────────────────
        p_body = ""
        if patient_out.patient_trend_phrase:
            p_body += (
                f"<p style='font-size:1.0rem;line-height:1.7;margin:0 0 12px 0;'>"
                f"<em>Your results show <strong>{patient_out.patient_trend_phrase}</strong>.</em></p>"
            )
        if patient_out.patient_explanation:
            p_body += (
                f"<div style='line-height:1.75;color:#4a3728;font-size:0.96rem;'>"
                f"{patient_out.patient_explanation}</div>"
            )
        if patient_out.patient_questions:
            qs = "".join(
                f"<li style='margin-bottom:4px;'>{q}</li>"
                for q in patient_out.patient_questions
            )
            p_body += (
                "<p style='margin:14px 0 5px;font-family:system-ui,sans-serif;font-size:0.78rem;"
                "font-weight:600;color:#7a6558;text-transform:uppercase;letter-spacing:.05em;'>"
                "Questions to ask your doctor</p>"
                f"<ul style='padding-left:18px;color:#4a3728;font-size:0.93rem;line-height:1.8;margin:0;'>{qs}</ul>"
            )
        if patient_out.patient_disclaimer:
            p_body += (
                "<div style='margin-top:16px;padding:10px 14px;border:1px solid #E8DDD6;"
                "border-radius:3px;background:#FDFAF7;'>"
                f"<p style='font-family:system-ui,sans-serif;font-size:0.77rem;font-style:italic;"
                f"color:#9a8578;margin:0;line-height:1.6;'>{patient_out.patient_disclaimer}</p>"
                "</div>"
            )
        patient_html = (
            "<div style='font-family:\'Source Serif 4\',serif;background:#FDFAF7;border:1px solid #E8DDD6;"
            "border-radius:4px;padding:22px 26px;box-shadow:0 1px 4px rgba(28,20,18,0.07);'>"
            "<h3 style='font-family:\'Playfair Display\',serif;font-size:1.1rem;font-weight:600;"
            "color:#C1622F;margin:0 0 14px;border-left:3px solid #C1622F;padding-left:10px;"
            "letter-spacing:.01em;'>Patient Summary</h3>"
            + p_body
            + "</div>"
        )

        # ── Clinician card ─────────────────────────────────────────────────
        # Confidence bar
        conf_val = clinician_out.clinician_confidence
        conf_pct_num = int((conf_val or 0) * 100)
        conf_label = f"{conf_val:.0%}" if conf_val is not None else "N/A"
        conf_bar = (
            "<div style='margin:0 0 14px;'>"
            "<div style='display:flex;align-items:baseline;gap:8px;margin-bottom:5px;'>"
            "<span style='font-family:system-ui,sans-serif;font-size:0.78rem;font-weight:600;"
            "color:#7a6558;text-transform:uppercase;letter-spacing:.04em;'>Confidence</span>"
            f"<span style='font-family:\'Playfair Display\',serif;font-size:1.12rem;"
            f"font-weight:700;color:#C1622F;'>{conf_label}</span>"
            "</div>"
            "<div style='height:5px;border-radius:3px;background:#E8DDD6;overflow:hidden;'>"
            f"<div style='height:100%;width:{conf_pct_num}%;background:#C1622F;border-radius:3px;'></div>"
            "</div></div>"
        )
        c_body = conf_bar
        if clinician_out.clinician_stewardship_flag:
            c_body += (
                "<div style='background:#fdf5f1;border-left:3px solid #C1622F;"
                "padding:10px 14px;margin:10px 0;border-radius:3px;'>"
                "<span style='font-family:system-ui,sans-serif;font-size:0.84rem;"
                "color:#C1622F;font-weight:600;'>⚠ Stewardship Alert</span>"
                "<p style='margin:4px 0 0;font-family:system-ui,sans-serif;font-size:0.82rem;"
                "color:#6b3320;'>Emerging resistance detected — antimicrobial stewardship review recommended.</p>"
                "</div>"
            )
        if clinician_out.clinician_resistance_detail:
            c_body += (
                "<div style='background:#FDFAF7;border-left:3px solid #E8DDD6;"
                "padding:10px 14px;margin:10px 0;border-radius:3px;'>"
                "<p style='margin:0 0 4px;font-family:system-ui,sans-serif;font-size:0.78rem;"
                "font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:#7a6558;'>"
                "Resistance Timeline</p>"
                f"<pre style='margin:0;font-size:12px;font-family:system-ui,monospace;"
                f"color:#4a3728;white-space:pre-wrap;'>{clinician_out.clinician_resistance_detail}</pre>"
                "</div>"
            )
        if clinician_out.clinician_interpretation:
            c_body += (
                f"<div style='line-height:1.75;color:#3d2b1f;font-size:0.96rem;margin-top:12px;'>"
                f"{clinician_out.clinician_interpretation}</div>"
            )
        if clinician_out.clinician_disclaimer:
            c_body += (
                "<p style='font-family:system-ui,sans-serif;font-style:italic;color:#9a8578;"
                "border-top:1px solid #E8DDD6;padding-top:10px;margin-top:18px;"
                f"font-size:0.77rem;line-height:1.6;'>{clinician_out.clinician_disclaimer}</p>"
            )
        clinician_html = (
            "<div style='font-family:\'Source Serif 4\',serif;background:#FDFAF7;border:1px solid #E8DDD6;"
            "border-radius:4px;padding:22px 26px;margin-top:14px;box-shadow:0 1px 4px rgba(28,20,18,0.07);'>"
            "<h3 style='font-family:\'Playfair Display\',serif;font-size:1.1rem;font-weight:600;"
            "color:#C1622F;margin:0 0 14px;border-left:3px solid #C1622F;padding-left:10px;"
            "letter-spacing:.01em;'>Clinical Interpretation</h3>"
            + c_body
            + "</div>"
        )

        return patient_html, clinician_html

    # ── Gradio Blocks ───────────────────────────────────────────────────────
    with gr.Blocks(
        title="CultureSense — Longitudinal Clinical Hypothesis Engine",
        theme=WARM_CLINICAL_THEME,
        css="""
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Serif+4:ital,wght@0,400;0,500;1,400&display=swap');

        .screen { padding: 8px 0; }
        .status-box {
            font-size: 0.85rem;
            line-height: 1.8;
            font-family: system-ui, sans-serif;
            color: #4a3728;
        }
        .error-banner {
            background: #fdf5f1;
            border: 1px solid #E8DDD6;
            border-left: 3px solid #C1622F;
            padding: 12px 16px;
            border-radius: 3px;
            margin: 10px 0;
            color: #6b3320;
            font-family: system-ui, sans-serif;
            font-size: 0.85rem;
        }
        /* UI chrome: inputs, labels, buttons — system-ui at small size */
        input, textarea, select, label, button {
            font-family: system-ui, sans-serif !important;
            font-size: 0.85rem !important;
        }
        /* Gradio tab labels */
        .tab-nav button { font-family: system-ui, sans-serif !important; font-size: 0.82rem !important; }
        /* Minimal shadows only */
        .gr-box, .gr-panel { box-shadow: 0 1px 4px rgba(28,20,18,0.07) !important; }
        .gr-button-primary { box-shadow: 0 1px 3px rgba(28,20,18,0.10) !important; }
        /* Section headings rendered by Gradio Markdown use Playfair Display */
        h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
        """,
    ) as demo:

        gr.Markdown(
            "# 🧫 CultureSense\n"
            "**Longitudinal Clinical Hypothesis Engine** — powered by MedGemma\n\n"
            "> Non-diagnostic. Always requires clinician review."
        )

        with gr.Tabs():

            # ================================================================
            # TAB A — Upload PDF (Extraction Agent)
            # ================================================================
            with gr.Tab("📄 Upload PDF", id="tab_upload"):

                # ── State ───────────────────────────────────────────────────
                state_reports = gr.State([])
                state_raw_blocks = gr.State([])

                # ── Screen 1: Upload ────────────────────────────────────────
                with gr.Column(visible=True, elem_classes="screen") as screen_upload:
                    gr.Markdown("### Step 1 — Upload your culture report PDFs")
                    gr.Markdown(
                        "Upload one or more PDF files. Each file may contain one or more "
                        "urine/stool culture reports."
                    )
                    pdf_upload = gr.File(
                        label="Culture Report PDFs",
                        file_types=[".pdf"],
                        file_count="multiple",
                    )
                    btn_process = gr.Button("⚙ Process PDFs", variant="primary")
                    status_html = gr.HTML(
                        value="", label="File Status", elem_classes="status-box"
                    )
                    with gr.Column(visible=False) as all_failed_panel:
                        gr.HTML(
                            '<div class="error-banner">'
                            "No urine or stool culture data was found in your uploaded documents. "
                            "Please try uploading again, or switch to manual entry."
                            "</div>"
                        )
                        with gr.Row():
                            btn_try_again = gr.Button("🔄 Try Again")
                            btn_to_manual_from_fail = gr.Button("✏ Enter Manually")

                # ── Screen 2: Review & Confirm ──────────────────────────────
                with gr.Column(visible=False, elem_classes="screen") as screen_confirm:
                    gr.Markdown("### Step 2 — Review & Confirm Extracted Records")
                    gr.Markdown(
                        "All cells are editable. Fields marked **⚠** were extracted with "
                        "low confidence — please verify against the raw text below."
                    )
                    truncation_warning_html = gr.HTML(value="")

                    confirm_table = gr.Dataframe(
                        headers=["Date", "Specimen", "Organism", "CFU/mL", "Resistance Markers"],
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=True,
                        wrap=True,
                        label="Extracted Culture Records",
                    )

                    with gr.Accordion(
                        "📋 Raw Extracted Text (for clinician verification)",
                        open=False,
                    ):
                        raw_box_0 = gr.Textbox(
                            label="Record 1", interactive=False, visible=False, lines=6
                        )
                        raw_box_1 = gr.Textbox(
                            label="Record 2", interactive=False, visible=False, lines=6
                        )
                        raw_box_2 = gr.Textbox(
                            label="Record 3", interactive=False, visible=False, lines=6
                        )

                    with gr.Row():
                        btn_confirm = gr.Button("✅ Confirm & Analyse", variant="primary")
                        btn_re_upload = gr.Button("↩ Edit & Re-upload")
                        btn_to_manual_from_confirm = gr.Button("✏ Enter Manually Instead")

                # ── Screen 3: Analysis Output ───────────────────────────────
                with gr.Column(visible=False, elem_classes="screen") as screen_output:
                    gr.Markdown("### Step 3 — Analysis Results")
                    output_patient_html = gr.HTML(value="")
                    output_clinician_html = gr.HTML(value="")
                    btn_start_over = gr.Button("🔄 Start Over")

                # ── Event: Process PDFs ─────────────────────────────────────
                def on_process_pdfs(files):
                    if not files:
                        return (
                            gr.update(),          # state_reports
                            gr.update(),          # state_raw_blocks
                            "<p style='color:#888'>No files uploaded.</p>",  # status_html
                            gr.update(visible=True),   # screen_upload
                            gr.update(visible=False),  # screen_confirm
                            gr.update(visible=False),  # screen_output
                            gr.update(visible=False),  # all_failed_panel
                            [],                        # confirm_table
                            "",                        # truncation_warning_html
                            gr.update(value="", visible=False),  # raw_box_0
                            gr.update(value="", visible=False),  # raw_box_1
                            gr.update(value="", visible=False),  # raw_box_2
                        )

                    reports, raw_blocks, statuses, trunc_warn = process_uploaded_pdfs(files)
                    status_combined = "".join(statuses)

                    if not reports:
                        # All files failed — stay on screen 1, show error panel
                        return (
                            [],
                            [],
                            status_combined,
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=True),
                            [],
                            "",
                            gr.update(value="", visible=False),
                            gr.update(value="", visible=False),
                            gr.update(value="", visible=False),
                        )

                    # Build dataframe rows
                    df_rows = reports_to_dataframe_rows(reports)

                    # Build raw text box updates (pre-created 3 boxes)
                    raw_updates = []
                    for i in range(MAX_RECORDS):
                        if i < len(raw_blocks):
                            raw_updates.append(
                                gr.update(
                                    value=raw_blocks[i],
                                    label=f"Record {i+1} — {reports[i].date}",
                                    visible=True,
                                )
                            )
                        else:
                            raw_updates.append(gr.update(value="", visible=False))

                    return (
                        reports,
                        raw_blocks,
                        status_combined,
                        gr.update(visible=False),  # hide screen_upload
                        gr.update(visible=True),   # show screen_confirm
                        gr.update(visible=False),  # hide screen_output
                        gr.update(visible=False),  # hide all_failed_panel
                        df_rows,
                        trunc_warn,
                        raw_updates[0],
                        raw_updates[1],
                        raw_updates[2],
                    )

                btn_process.click(
                    fn=on_process_pdfs,
                    inputs=[pdf_upload],
                    outputs=[
                        state_reports,
                        state_raw_blocks,
                        status_html,
                        screen_upload,
                        screen_confirm,
                        screen_output,
                        all_failed_panel,
                        confirm_table,
                        truncation_warning_html,
                        raw_box_0,
                        raw_box_1,
                        raw_box_2,
                    ],
                )

                # ── Event: Confirm & Analyse ────────────────────────────────
                def on_confirm(table_data):
                    if table_data is None or len(table_data) == 0:
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            "<p style='color:#c0392b'>No records to analyse.</p>",
                            "",
                        )

                    # Convert edited table rows back to CultureReport objects
                    confirmed_reports = []
                    for row in table_data:
                        try:
                            confirmed_reports.append(dataframe_row_to_culture_report(row))
                        except Exception:
                            pass

                    if not confirmed_reports:
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            "<p style='color:#c0392b'>Could not parse records.</p>",
                            "",
                        )

                    try:
                        patient_out, clinician_out = run_pipeline(confirmed_reports)
                        patient_html, clinician_html = format_output_html(patient_out, clinician_out)
                    except Exception as e:
                        patient_html = f"<p style='color:#c0392b'>Analysis error: {e}</p>"
                        clinician_html = ""

                    return (
                        gr.update(visible=False),  # hide screen_confirm
                        gr.update(visible=False),  # hide screen_upload
                        gr.update(visible=True),   # show screen_output
                        patient_html,
                        clinician_html,
                    )

                btn_confirm.click(
                    fn=on_confirm,
                    inputs=[confirm_table],
                    outputs=[
                        screen_confirm,
                        screen_upload,
                        screen_output,
                        output_patient_html,
                        output_clinician_html,
                    ],
                )

                # ── Event: Edit & Re-upload ─────────────────────────────────
                def on_re_upload():
                    return (
                        gr.update(visible=True),   # show screen_upload
                        gr.update(visible=False),  # hide screen_confirm
                        gr.update(visible=False),  # hide screen_output
                        gr.update(visible=False),  # hide all_failed_panel
                        [],                        # clear state_reports
                        [],                        # clear state_raw_blocks
                        "",                        # clear status_html
                    )

                btn_re_upload.click(
                    fn=on_re_upload,
                    inputs=[],
                    outputs=[
                        screen_upload,
                        screen_confirm,
                        screen_output,
                        all_failed_panel,
                        state_reports,
                        state_raw_blocks,
                        status_html,
                    ],
                )

                # ── Event: Try Again ────────────────────────────────────────
                btn_try_again.click(
                    fn=lambda: (
                        gr.update(visible=False),
                        "",
                    ),
                    inputs=[],
                    outputs=[all_failed_panel, status_html],
                )

                # ── Event: Start Over ───────────────────────────────────────
                def on_start_over():
                    return (
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        [],
                        [],
                        "",
                        "",
                        "",
                    )

                btn_start_over.click(
                    fn=on_start_over,
                    inputs=[],
                    outputs=[
                        screen_upload,
                        screen_confirm,
                        screen_output,
                        all_failed_panel,
                        state_reports,
                        state_raw_blocks,
                        status_html,
                        output_patient_html,
                        output_clinician_html,
                    ],
                )

            # ================================================================
            # TAB B — Enter Manually (existing flow — zero changes to pipeline)
            # ================================================================
            with gr.Tab("✏ Enter Manually", id="tab_manual"):
                gr.Markdown("### Manual Entry")
                gr.Markdown(
                    "Paste each culture report into a separate text box below. "
                    "The system will extract structured data and run the full analysis."
                )

                with gr.Row():
                    with gr.Column():
                        report_1 = gr.Textbox(
                            label="Culture Report 1 (required)",
                            placeholder=(
                                "Date: 2026-01-01\n"
                                "Specimen: urine\n"
                                "Organism: E. coli\n"
                                "CFU/mL: 120,000\n"
                                "Resistance: ESBL"
                            ),
                            lines=8,
                        )
                        report_2 = gr.Textbox(
                            label="Culture Report 2 (required)",
                            placeholder="Date: 2026-01-10\n...",
                            lines=8,
                        )
                        report_3 = gr.Textbox(
                            label="Culture Report 3 (optional)",
                            placeholder="Date: 2026-01-20\n...",
                            lines=8,
                        )

                btn_analyse_manual = gr.Button("🔬 Analyse Reports", variant="primary")
                manual_error = gr.HTML(value="")
                manual_patient_html = gr.HTML(value="")
                manual_clinician_html = gr.HTML(value="")

                def on_analyse_manual(r1, r2, r3):
                    texts = [t for t in [r1, r2, r3] if t and t.strip()]
                    if len(texts) < 2:
                        return (
                            "<p style='color:#c0392b'>Please enter at least 2 culture reports.</p>",
                            "",
                            "",
                        )

                    reports = []
                    for text in texts:
                        try:
                            report = extract_structured_data(text)
                            # Ensure raw_text is cleared before passing downstream
                            reports.append(CultureReport(
                                date=report.date,
                                organism=report.organism,
                                cfu=report.cfu,
                                resistance_markers=report.resistance_markers,
                                specimen_type=report.specimen_type,
                                contamination_flag=report.contamination_flag,
                                raw_text="",  # SAFETY: never forwarded to MedGemma
                            ))
                        except ExtractionError as e:
                            return (
                                f"<p style='color:#c0392b'>Extraction error: {e}</p>",
                                "",
                                "",
                            )

                    try:
                        patient_out, clinician_out = run_pipeline(reports)
                        patient_html, clinician_html = format_output_html(patient_out, clinician_out)
                        return "", patient_html, clinician_html
                    except Exception as e:
                        return (
                            f"<p style='color:#c0392b'>Analysis error: {e}</p>",
                            "",
                            "",
                        )

                btn_analyse_manual.click(
                    fn=on_analyse_manual,
                    inputs=[report_1, report_2, report_3],
                    outputs=[manual_error, manual_patient_html, manual_clinician_html],
                )

    return demo
