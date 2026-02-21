"""
CultureSense Extraction Agent (Cell J)
PDF upload flow: Docling → extract_structured_data() → Gradio UI

Three-screen state machine:
  Screen 1 — Upload PDFs
  Screen 2 — Review & Confirm extracted records (editable table)
  Screen 3 — Analysis output (existing pipeline, zero changes)

Tab B (manual entry) is the existing flow — zero modifications.
"""

import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from data_models import CultureReport
from extraction import ExtractionError, debug_extraction, extract_structured_data
from hypothesis import generate_hypothesis
from medgemma import call_medgemma
from renderer import render_clinician_output, render_patient_output
from rules import RULES
from trend import analyze_trend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RECORDS = 3
_WARN_PREFIX = "⚠ "

# ---------------------------------------------------------------------------
# Theme Definition — Warm Classical Medical Journal Aesthetic
# Base: warm white #FDFAF7 | Accent: burnt sienna #C1622F
# Headings: Playfair Display | Body: Source Serif 4 | UI: system-ui
# ---------------------------------------------------------------------------

WARM_CLINICAL_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="stone",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Source Serif 4"), "Georgia", "serif"],
    font_mono=[gr.themes.GoogleFont("Source Code Pro"), "monospace"],
).set(
    # Warm white background
    body_background_fill="#FDFAF7",
    background_fill_primary="#FDFAF7",
    background_fill_secondary="#F5F0EB",

    # Warm gray borders
    border_color_primary="#E8DDD6",
    block_border_color="#E8DDD6",

    # Burnt sienna accent for buttons
    button_primary_background_fill="#C1622F",
    button_primary_background_fill_hover="#a85228",
    button_primary_text_color="#FDFAF7",

    # Subtle shadows only
    block_shadow="0 1px 4px rgba(28,20,18,0.07)",
)


# ---------------------------------------------------------------------------
# 1. Docling PDF processor with enhanced error handling
# ---------------------------------------------------------------------------


def process_pdf_file(pdf_path: str) -> Tuple[str, str, str]:
    """
    Parse a single PDF with Docling.

    Returns:
        (markdown_text, status_html, debug_info)
        - On success: (markdown, "", debug_info)
        - On parse failure: ("", "<red status>", error_details)
    """
    debug_info = f"Processing: {Path(pdf_path).name}\n"

    try:
        from docling.document_converter import DocumentConverter

        debug_info += "✓ Docling imported successfully\n"

        converter = DocumentConverter()
        debug_info += "✓ DocumentConverter created\n"

        start_time = time.time()
        result = converter.convert(pdf_path)
        elapsed = time.time() - start_time
        debug_info += f"✓ PDF converted in {elapsed:.1f}s\n"

        markdown_text = result.document.export_to_markdown()
        debug_info += f"✓ Markdown exported ({len(markdown_text)} chars)\n"

        # Preview first 500 chars for debugging
        preview = markdown_text[:500].replace("\n", " ")
        debug_info += f"Preview: {preview}...\n"

        return markdown_text, "", debug_info

    except ImportError as e:
        error_msg = f"✗ Docling not installed: {e}"
        debug_info += error_msg + "\n"
        return (
            "",
            f'<span style="color:#c0392b">{error_msg}</span>',
            debug_info,
        )
    except Exception as e:
        error_msg = f"✗ PDF processing failed: {type(e).__name__}: {str(e)[:100]}"
        debug_info += error_msg + "\n"
        return (
            "",
            f'<span style="color:#c0392b">{error_msg}</span>',
            debug_info,
        )


# ---------------------------------------------------------------------------
# 2. Multi-report splitter (unchanged)
# ---------------------------------------------------------------------------


def _split_into_report_blocks(markdown_text: str) -> List[str]:
    """
    Attempt to split a multi-report markdown document into individual report blocks.

    Heuristic: split on "MICROBIOLOGY REPORT" headings, then attach dates
    in the order they appear in the markdown.
    Falls back to returning the whole text as one block.
    """
    import re

    # Try splitting on "---" or "===" separators
    blocks = re.split(r"\n(?:---+|===+)\n", markdown_text)
    if len(blocks) > 1:
        return [b.strip() for b in blocks if b.strip()]

    # Find all "Collected:" dates in order
    collected_pattern = re.compile(r"Collected:\s*(\d{4}-\d{2}-\d{2})")
    collected_dates = collected_pattern.findall(markdown_text)

    # Try splitting on MICROBIOLOGY REPORT headings
    pattern = r"\n(?=#{1,2}\s*MICROBIOLOGY\s+REPORT\b)"
    parts = re.split(pattern, markdown_text, flags=re.IGNORECASE)

    if len(parts) > 1:
        result = []
        # Skip the first part (header info)
        for i in range(1, len(parts)):
            part = parts[i].strip()
            if not part:
                continue

            # Assign date by index (in order of appearance)
            date_idx = i - 1
            if date_idx < len(collected_dates):
                part = f"Collected: {collected_dates[date_idx]}\n\n" + part

            result.append(part)
        return result

    # Single block - do NOT split on arbitrary H1/H2 headers as they are
    # section headers within a report (e.g., "## SPECIMEN INFORMATION", "## CULTURE RESULT")
    # not report boundaries. Splitting on them breaks extraction of single reports
    # that have multiple sections (like SetD which has CBC, metabolic panel, and urine culture).
    return [markdown_text.strip()] if markdown_text.strip() else []


def _is_low_confidence(report: CultureReport) -> bool:
    """Return True if any field looks suspiciously generic."""
    return (
        report.organism == "unknown"
        or report.date == "unknown"
        or (report.cfu == 0 and "no growth" not in report.raw_text.lower())
    )


# ---------------------------------------------------------------------------
# 3. DataFrame helpers (unchanged)
# ---------------------------------------------------------------------------


def reports_to_dataframe_rows(reports: List[CultureReport]) -> List[List[str]]:
    """Convert CultureReport list to list of list strings for gr.Dataframe."""
    rows = []
    for r in reports:
        warn = _WARN_PREFIX if _is_low_confidence(r) else ""
        rows.append(
            [
                f"{warn}{r.date}",
                r.specimen_type,
                r.organism,
                str(r.cfu),
                ", ".join(r.resistance_markers) if r.resistance_markers else "—",
            ]
        )
    return rows


def dataframe_row_to_culture_report(row: List[str]) -> CultureReport:
    """Convert a single Dataframe row (list of strings) back to CultureReport."""
    date_str = row[0].replace(_WARN_PREFIX, "").strip()
    specimen = row[1].strip()
    organism = row[2].strip()
    cfu_str = row[3].replace(",", "").strip()
    resistance_str = row[4].strip()

    try:
        cfu = int(cfu_str)
    except ValueError:
        cfu = 0

    resistance_markers = (
        [m.strip() for m in resistance_str.split(",") if m.strip() != "—"]
        if resistance_str != "—"
        else []
    )

    return CultureReport(
        date=date_str,
        organism=organism,
        cfu=cfu,
        resistance_markers=resistance_markers,
        specimen_type=specimen,
        contamination_flag=any(
            term in organism.lower() for term in RULES["contamination_terms"]
        ),
        raw_text="",  # Not needed for downstream pipeline
    )


# ---------------------------------------------------------------------------
# 4. PDF batch processor with enhanced error handling and debug output
# ---------------------------------------------------------------------------


def process_uploaded_pdfs(
    files: List,
) -> Tuple[List[CultureReport], List[str], List[str], str, str]:
    """
    Process a list of uploaded PDF file objects from gr.File.

    Returns:
        (reports, raw_text_blocks, per_file_statuses, truncation_warning, debug_log)
        - reports: deduplicated, sorted, max MAX_RECORDS CultureReport list
        - raw_text_blocks: one markdown string per report (for clinician accordion)
        - per_file_statuses: one HTML status string per uploaded file
        - truncation_warning: non-empty string if records were truncated
        - debug_log: detailed processing log for troubleshooting
    """
    debug_log = "=== PDF Processing Debug Log ===\n\n"

    if not files:
        debug_log += "No files provided\n"
        return [], [], [], "", debug_log

    all_reports: List[CultureReport] = []
    all_raw_blocks: List[str] = []
    per_file_statuses: List[str] = []

    debug_log += f"Processing {len(files)} file(s)...\n\n"

    for i, f in enumerate(files, 1):
        # Gradio passes file objects with a .name attribute (temp path)
        pdf_path = f.name if hasattr(f, "name") else str(f)
        filename = Path(pdf_path).name

        debug_log += f"--- File {i}/{len(files)}: {filename} ---\n"

        markdown_text, parse_error, file_debug = process_pdf_file(pdf_path)
        debug_log += file_debug

        if parse_error:
            per_file_statuses.append(
                f'<div style="margin:4px 0"><b>{filename}</b> — {parse_error}</div>'
            )
            debug_log += f"✗ Skipped due to parse error\n\n"
            continue

        # Try to extract culture records from the markdown
        # extract_structured_data() handles one report block at a time.
        # For multi-report PDFs, split on common section delimiters.
        report_blocks = _split_into_report_blocks(markdown_text)
        debug_log += f"✓ Split into {len(report_blocks)} block(s)\n"

        file_reports: List[CultureReport] = []

        for block_idx, block in enumerate(report_blocks, 1):
            debug_log += f"\n  Block {block_idx}:\n"
            try:
                # Debug extraction
                debug_result = debug_extraction(block, f"Block {block_idx}")
                debug_log += f"    Organism: {debug_result['organism']}\n"
                debug_log += (
                    f"    CFU: {debug_result['cfu']} (ok={debug_result['cfu_ok']})\n"
                )
                debug_log += f"    Specimen: {debug_result['specimen']}\n"
                debug_log += f"    Date: {debug_result['date']}\n"

                report = extract_structured_data(block)
                debug_log += f"    ✓ Extraction successful\n"

                # Only keep urine/stool specimens
                if report.specimen_type in ("urine", "stool"):
                    debug_log += (
                        f"    ✓ Specimen type '{report.specimen_type}' accepted\n"
                    )
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
                else:
                    debug_log += f"    ✗ Specimen type '{report.specimen_type}' rejected (not urine/stool)\n"

            except ExtractionError as e:
                debug_log += f"    ✗ ExtractionError: {e}\n"
                pass  # block had no parseable culture data
            except Exception as e:
                debug_log += f"    ✗ Unexpected error: {type(e).__name__}: {e}\n"
                pass

        if not file_reports:
            per_file_statuses.append(
                f'<div style="margin:4px 0"><b>{filename}</b> — '
                f'<span style="color:#e67e22">⚠ No urine or stool culture data found in this file</span></div>'
            )
            debug_log += f"\n✗ No valid culture records found in {filename}\n\n"
        else:
            count = len(file_reports)
            per_file_statuses.append(
                f'<div style="margin:4px 0"><b>{filename}</b> — '
                f'<span style="color:#27ae60">✓ {count} record{"s" if count != 1 else ""} found</span></div>'
            )
            all_reports.extend(file_reports)
            all_raw_blocks.extend(r.raw_text for r in file_reports)
            debug_log += f"\n✓ Extracted {count} record(s) from {filename}\n\n"

    if not all_reports:
        debug_log += "=== RESULT: No valid reports found ===\n"
        return [], [], per_file_statuses, "", debug_log

    # Sort chronologically
    debug_log += f"Sorting {len(all_reports)} report(s) chronologically...\n"
    combined = sorted(zip(all_reports, all_raw_blocks), key=lambda pair: pair[0].date)
    all_reports = [p[0] for p in combined]
    all_raw_blocks = [p[1] for p in combined]

    # Deduplicate: same (date, organism, cfu) → keep first
    seen: set = set()
    deduped_reports: List[CultureReport] = []
    deduped_blocks: List[str] = []
    for report, block in zip(all_reports, all_raw_blocks):
        key = (report.date, report.organism, report.cfu)
        if key in seen:
            debug_log += f"⚠ Duplicate record skipped: {key}\n"
            warnings.warn(f"Duplicate record skipped: {key}", UserWarning, stacklevel=2)
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
            f"⚠ {total} records were extracted. Only the {MAX_RECORDS} most recent are shown "
            f"(the pipeline supports up to {MAX_RECORDS} reports).</div>"
        )
        debug_log += f"⚠ Truncated from {total} to {MAX_RECORDS} most recent records\n"

    debug_log += f"\n=== RESULT: Returning {len(deduped_reports)} report(s) ===\n"
    for i, r in enumerate(deduped_reports, 1):
        debug_log += (
            f"  {i}. {r.date} | {r.specimen_type} | {r.organism} | {r.cfu} CFU\n"
        )

    return (
        deduped_reports,
        deduped_blocks,
        per_file_statuses,
        truncation_warning,
        debug_log,
    )


# ---------------------------------------------------------------------------
# 5. Gradio UI builder with loading indicators
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
        patient_response = call_medgemma(
            trend, hypothesis, "patient", model, tokenizer, is_stub
        )
        clinician_response = call_medgemma(
            trend, hypothesis, "clinician", model, tokenizer, is_stub
        )
        patient_out = render_patient_output(trend, hypothesis, patient_response)
        clinician_out = render_clinician_output(trend, hypothesis, clinician_response)
        return patient_out, clinician_out

    def format_output_html(patient_out, clinician_out) -> Tuple[str, str]:
        """Convert FormattedOutput objects to display HTML — warm classical styling."""
        # ── Patient card ───────────────────────────────────────────────────
        p_body = ""
        if patient_out.patient_trend_phrase:
            p_body += (
                f"<p style='font-size:1.0rem;line-height:1.7;margin:0 0 12px 0;'>"
                f"<em>Your results show <strong>{patient_out.patient_trend_phrase}</strong>.</em></p>"
            )
        if patient_out.patient_explanation:
            p_body += (
                f"<div style='line-height:1.75;font-size:0.96rem;'>"
                f"{patient_out.patient_explanation}</div>"
            )
        if patient_out.patient_questions:
            qs = "".join(
                f"<li style='margin-bottom:4px;'>{q}</li>"
                for q in patient_out.patient_questions
            )
            p_body += (
                "<p style='margin:16px 0 8px;font-size:0.75rem;font-weight:600;"
                "text-transform:uppercase;letter-spacing:0.05em;color:#7A6558;'>"
                "Questions to ask your doctor</p>"
                f"<ul style='padding-left:20px;font-size:0.93rem;line-height:1.8;margin:0;'>{qs}</ul>"
            )
        if patient_out.patient_disclaimer:
            p_body += (
                "<div style='margin-top:16px;padding:12px 14px;border:1px solid #E8DDD6;"
                "border-radius:6px;background:#EDE7E0;'>"
                f"<p style='font-size:0.77rem;font-style:italic;color:#5D4037;margin:0;line-height:1.6;'>"
                f"{patient_out.patient_disclaimer}</p></div>"
            )
        patient_html = (
            "<div class='output-card'>"
            "<h3>📋 Patient Summary</h3>" + p_body + "</div>"
        )

        # ── Clinician card ─────────────────────────────────────────────────
        conf_val = clinician_out.clinician_confidence
        conf_pct_num = int((conf_val or 0) * 100)
        conf_label = f"{conf_val:.0%}" if conf_val is not None else "N/A"
        conf_bar = (
            "<div style='margin:0 0 16px;'>"
            "<div style='display:flex;align-items:baseline;gap:8px;margin-bottom:6px;'>"
            "<span style='font-size:0.75rem;font-weight:600;text-transform:uppercase;"
            "letter-spacing:0.04em;color:#7A6558;'>Confidence</span>"
            f"<span style='font-size:1.25rem;font-weight:700;color:#C1622F;'>{conf_label}</span>"
            "</div>"
            "<div style='height:6px;border-radius:3px;background:#E8DDD6;overflow:hidden;'>"
            f"<div style='height:100%;width:{conf_pct_num}%;background:#C1622F;border-radius:3px;'></div>"
            "</div></div>"
        )
        c_body = conf_bar
        if clinician_out.clinician_stewardship_flag:
            c_body += (
                "<div style='background:#FDF5F1;border-left:3px solid #C1622F;"
                "padding:12px 14px;margin:12px 0;border-radius:6px;'>"
                "<span style='font-size:0.85rem;font-weight:600;color:#8B4513;'>⚠ Stewardship Alert</span>"
                "<p style='margin:4px 0 0;font-size:0.82rem;color:#5D4037;'>"
                "Emerging resistance detected — antimicrobial stewardship review recommended.</p>"
                "</div>"
            )
        if clinician_out.clinician_resistance_detail:
            c_body += (
                "<div style='background:#EDE7E0;border-left:3px solid #D4A574;"
                "padding:12px 14px;margin:12px 0;border-radius:6px;'>"
                "<p style='margin:0 0 6px;font-size:0.75rem;font-weight:600;text-transform:uppercase;"
                "letter-spacing:0.04em;color:#7A6558;'>Resistance Timeline</p>"
                f"<pre style='margin:0;font-size:12px;font-family:\"Source Code Pro\",monospace;white-space:pre-wrap;color:#4A3728;'>"
                f"{clinician_out.clinician_resistance_detail}</pre></div>"
            )
        if clinician_out.clinician_interpretation:
            c_body += (
                f"<div style='line-height:1.75;font-size:0.96rem;margin-top:12px;'>"
                f"{clinician_out.clinician_interpretation}</div>"
            )
        if clinician_out.clinician_disclaimer:
            c_body += (
                "<p style='font-style:italic;color:#7A6558;border-top:1px solid #E8DDD6;"
                "padding-top:12px;margin-top:18px;font-size:0.77rem;line-height:1.6;'>"
                f"{clinician_out.clinician_disclaimer}</p>"
            )
        clinician_html = (
            "<div class='output-card'>"
            "<h3>🩺 Clinical Interpretation</h3>" + c_body + "</div>"
        )

        return patient_html, clinician_html

    # ── Build UI ────────────────────────────────────────────────────────────
    with gr.Blocks(
        theme=WARM_CLINICAL_THEME,
        css="""
        /* Import Playfair Display for headings */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&display=swap');

        .screen { min-height: 60vh; }

        /* Status box - warm paper texture */
        .status-box {
            min-height: 40px;
            border: 1px solid #E8DDD6;
            border-radius: 4px;
            padding: 12px 16px;
            background: #FDFAF7;
            font-family: system-ui, sans-serif;
            font-size: 0.875rem;
        }

        /* Error banner - muted warm tones */
        .error-banner {
            background: #FDF5F1;
            border-left: 3px solid #C1622F;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 4px;
            color: #5D4037;
            font-family: system-ui, sans-serif;
            font-size: 0.875rem;
        }

        /* Loading spinner */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #E8DDD6;
            border-top: 2px solid #C1622F;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* Output cards - medical journal style */
        .output-card {
            border: 1px solid #E8DDD6;
            border-radius: 4px;
            padding: 22px 26px;
            background: #FDFAF7;
            margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(28,20,18,0.07);
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 0.96rem;
            line-height: 1.75;
            color: #4A3728;
        }
        .output-card h3 {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #C1622F;
            margin: 0 0 14px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #E8DDD6;
            letter-spacing: 0.01em;
        }

        /* PDF count header */
        .pdf-count-header {
            margin-bottom: 8px;
            padding: 10px 14px;
            background: #F5F0EB;
            border-radius: 4px;
            font-family: system-ui, sans-serif;
            font-weight: 500;
            font-size: 0.875rem;
            color: #5D4037;
        }

        /* File status items */
        .file-status {
            padding: 6px 0;
            border-bottom: 1px solid #EDE7E0;
            font-family: system-ui, sans-serif;
            font-size: 0.875rem;
        }
        .file-status:last-child { border-bottom: none; }

        /* Labels and UI chrome */
        label, .gradio-label {
            font-family: system-ui, sans-serif !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            color: #7A6558 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Section headings */
        h3.section-heading {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #C1622F;
            border-left: 3px solid #C1622F;
            padding-left: 10px;
            margin: 0 0 14px 0;
            letter-spacing: 0.01em;
        }
    """,
    ) as demo:
        gr.Markdown("# 🧫 CultureSense — Longitudinal Clinical Hypothesis Engine")
        gr.Markdown(
            "Upload 2–3 sequential urine or stool culture reports to generate a trend analysis and clinical hypothesis."
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

                    with gr.Row():
                        btn_process = gr.Button("⚙ Process PDFs", variant="primary")
                        btn_process_loading = gr.Button(
                            "⏳ Processing...",
                            variant="primary",
                            interactive=False,
                            visible=False,
                        )

                    status_html = gr.HTML(
                        value="", label="File Status", elem_classes="status-box"
                    )

                    # Loading indicator
                    loading_html = gr.HTML(
                        value="",
                        visible=False,
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

                    # Debug output (collapsed by default)
                    with gr.Accordion(
                        "🔍 Debug Output (click to expand if processing fails)",
                        open=False,
                    ):
                        debug_output = gr.Textbox(
                            label="Processing Log",
                            interactive=False,
                            lines=20,
                            value="",
                        )

                # ── Screen 2: Review & Confirm ──────────────────────────────
                with gr.Column(visible=False, elem_classes="screen") as screen_confirm:
                    gr.Markdown("### Step 2 — Review & Confirm Extracted Records")
                    gr.Markdown(
                        "All cells are editable. Fields marked **⚠** were extracted with "
                        "low confidence — please verify against the raw text below."
                    )
                    truncation_warning_html = gr.HTML(value="")

                    confirm_table = gr.Dataframe(
                        headers=[
                            "Date",
                            "Specimen",
                            "Organism",
                            "CFU/mL",
                            "Resistance Markers",
                        ],
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
                        btn_confirm = gr.Button(
                            "✅ Confirm & Analyse", variant="primary"
                        )
                        btn_re_upload = gr.Button("↩ Edit & Re-upload")
                        btn_to_manual_from_confirm = gr.Button(
                            "✏ Enter Manually Instead"
                        )

                # ── Screen 3: Analysis Output ───────────────────────────────
                with gr.Column(visible=False, elem_classes="screen") as screen_output:
                    gr.Markdown("### Step 3 — Analysis Results")
                    output_patient_html = gr.HTML(value="")
                    output_clinician_html = gr.HTML(value="")
                    btn_start_over = gr.Button("🔄 Start Over")

                # ── Event: Process PDFs ─────────────────────────────────────
                def on_process_pdfs_start(files):
                    """Show loading state immediately when button is clicked."""
                    if not files:
                        return (
                            gr.update(visible=True),  # btn_process
                            gr.update(visible=False),  # btn_process_loading
                            gr.update(
                                value="<p style='color:#888'>No files uploaded.</p>",
                                visible=True,
                            ),
                            gr.update(visible=False),  # loading_html
                        )

                    # Show loading state
                    loading_msg = (
                        '<div style="padding:12px;background:#fff3cd;border:1px solid #ffc107;border-radius:4px;">'
                        '<span class="loading-spinner"></span>'
                        "<strong>Processing PDFs...</strong> This may take 30-60 seconds per file. "
                        "Docling is extracting text from your PDFs."
                        "</div>"
                    )

                    return (
                        gr.update(visible=False),  # btn_process
                        gr.update(visible=True),  # btn_process_loading
                        gr.update(value=loading_msg, visible=True),  # status_html
                        gr.update(visible=True),  # loading_html
                    )

                def on_process_pdfs(files):
                    """Actually process the PDFs after loading state is shown."""
                    if not files:
                        return (
                            [],  # state_reports
                            [],  # state_raw_blocks
                            "<p style='color:#888'>No files uploaded.</p>",  # status_html
                            gr.update(visible=True),  # screen_upload
                            gr.update(visible=False),  # screen_confirm
                            gr.update(visible=False),  # screen_output
                            gr.update(visible=False),  # all_failed_panel
                            [],  # confirm_table
                            "",  # truncation_warning_html
                            gr.update(value="", visible=False),  # raw_box_0
                            gr.update(value="", visible=False),  # raw_box_1
                            gr.update(value="", visible=False),  # raw_box_2
                            "",  # debug_output
                            gr.update(visible=True),  # btn_process
                            gr.update(visible=False),  # btn_process_loading
                            gr.update(visible=False),  # loading_html
                        )

                    reports, raw_blocks, statuses, trunc_warn, debug_log = (
                        process_uploaded_pdfs(files)
                    )
                    # Add header showing total PDFs uploaded
                    pdf_count = len(files) if files else 0
                    status_header = (
                        f'<div style="margin-bottom:8px;padding:8px 12px;background:#f0f0f0;'
                        f'border-radius:4px;font-weight:500;">'
                        f'📄 {pdf_count} PDF{"s" if pdf_count != 1 else ""} uploaded</div>'
                    )
                    status_combined = status_header + "".join(statuses)

                    if not reports:
                        # All files failed — stay on screen 1, show error panel
                        error_msg = (
                            status_header +
                            '<div style="padding:12px;background:#f8d7da;border:1px solid #f5c6cb;border-radius:4px;color:#721c24;">'
                            "<strong>✗ No valid culture data found</strong><br>"
                            "Please check the debug output below for details."
                            "</div>"
                        )
                        return (
                            [],
                            [],
                            error_msg,
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=True),
                            [],
                            "",
                            gr.update(value="", visible=False),
                            gr.update(value="", visible=False),
                            gr.update(value="", visible=False),
                            debug_log,  # Show debug log
                            gr.update(visible=True),  # btn_process
                            gr.update(visible=False),  # btn_process_loading
                            gr.update(visible=False),  # loading_html
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
                                    label=f"Record {i + 1} — {reports[i].date}",
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
                        gr.update(visible=True),  # show screen_confirm
                        gr.update(visible=False),  # hide screen_output
                        gr.update(visible=False),  # hide all_failed_panel
                        df_rows,
                        trunc_warn,
                        raw_updates[0],
                        raw_updates[1],
                        raw_updates[2],
                        debug_log,  # Store debug log
                        gr.update(visible=True),  # btn_process
                        gr.update(visible=False),  # btn_process_loading
                        gr.update(visible=False),  # loading_html
                    )

                # Chain the events: first show loading, then process
                btn_process.click(
                    fn=on_process_pdfs_start,
                    inputs=[pdf_upload],
                    outputs=[
                        btn_process,
                        btn_process_loading,
                        status_html,
                        loading_html,
                    ],
                ).then(
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
                        debug_output,
                        btn_process,
                        btn_process_loading,
                        loading_html,
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
                            confirmed_reports.append(
                                dataframe_row_to_culture_report(row)
                            )
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
                        patient_html, clinician_html = format_output_html(
                            patient_out, clinician_out
                        )
                    except Exception as e:
                        patient_html = (
                            f"<p style='color:#c0392b'>Analysis error: {e}</p>"
                        )
                        clinician_html = ""

                    return (
                        gr.update(visible=False),  # hide screen_confirm
                        gr.update(visible=False),  # hide screen_upload
                        gr.update(visible=True),  # show screen_output
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
                        gr.update(visible=True),  # show screen_upload
                        gr.update(visible=False),  # hide screen_confirm
                        gr.update(visible=False),  # hide screen_output
                        gr.update(visible=False),  # hide all_failed_panel
                        [],  # clear state_reports
                        [],  # clear state_raw_blocks
                        "",  # clear status_html
                        "",  # clear debug_output
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
                        debug_output,
                    ],
                )

                # ── Event: Try Again (from fail panel) ──────────────────────
                btn_try_again.click(
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
                        debug_output,
                    ],
                )

                # ── Event: Start Over ───────────────────────────────────────
                def on_start_over():
                    return (
                        gr.update(visible=True),  # show screen_upload
                        gr.update(visible=False),  # hide screen_confirm
                        gr.update(visible=False),  # hide screen_output
                        gr.update(visible=False),  # hide all_failed_panel
                        [],  # clear state_reports
                        [],  # clear state_raw_blocks
                        "",  # clear status_html
                        None,  # clear pdf_upload
                        "",  # clear debug_output
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
                        pdf_upload,
                        debug_output,
                    ],
                )

                # ── Event: Switch to Manual Entry ───────────────────────────
                def switch_to_manual():
                    return (
                        gr.update(visible=False),  # hide upload screen
                        gr.update(visible=False),  # hide confirm screen
                        gr.update(visible=False),  # hide output screen
                        gr.update(visible=False),  # hide fail panel
                        gr.update(value="manual"),  # switch tab
                    )

                btn_to_manual_from_fail.click(
                    fn=switch_to_manual,
                    inputs=[],
                    outputs=[
                        screen_upload,
                        screen_confirm,
                        screen_output,
                        all_failed_panel,
                        gr.State("manual"),  # dummy, will be replaced by tab selection
                    ],
                )

                btn_to_manual_from_confirm.click(
                    fn=switch_to_manual,
                    inputs=[],
                    outputs=[
                        screen_upload,
                        screen_confirm,
                        screen_output,
                        all_failed_panel,
                        gr.State("manual"),
                    ],
                )

            # ================================================================
            # TAB B — Manual Entry (unchanged from original)
            # ================================================================
            with gr.Tab("✏ Enter Manually", id="tab_manual"):
                gr.Markdown("### Paste culture report text directly")
                gr.Markdown(
                    "Paste 2–3 sequential culture reports. "
                    "The pipeline will extract structured data, analyse trends, and generate hypotheses."
                )

                manual_input = gr.Textbox(
                    label="Culture Reports (2–3 sequential)",
                    placeholder="Paste report text here...",
                    lines=12,
                )
                btn_analyse_manual = gr.Button("🔬 Analyse", variant="primary")
                manual_output_patient = gr.HTML()
                manual_output_clinician = gr.HTML()

                def on_analyse_manual(text):
                    if not text or len(text.strip()) < 20:
                        return (
                            "<p style='color:#c0392b'>Please paste at least one full report.</p>",
                            "",
                        )

                    # Split by double newlines to get separate reports
                    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
                    reports = []
                    for block in blocks:
                        try:
                            r = extract_structured_data(block)
                            reports.append(r)
                        except Exception:
                            pass

                    if len(reports) < 1:
                        return (
                            "<p style='color:#c0392b'>Could not extract data from pasted text. "
                            "Check format includes Date, Organism, and CFU/mL.</p>",
                            "",
                        )

                    try:
                        patient_out, clinician_out = run_pipeline(reports)
                        patient_html, clinician_html = format_output_html(
                            patient_out, clinician_out
                        )
                    except Exception as e:
                        patient_html = (
                            f"<p style='color:#c0392b'>Analysis error: {e}</p>"
                        )
                        clinician_html = ""

                    return patient_html, clinician_html

                btn_analyse_manual.click(
                    fn=on_analyse_manual,
                    inputs=[manual_input],
                    outputs=[manual_output_patient, manual_output_clinician],
                )

    return demo
