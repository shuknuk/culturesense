"""
CultureSense Extraction Agent (Cell J)
PDF upload flow: Docling → extract_structured_data() → Gradio UI

Three-screen state machine:
  Screen 1 — Upload PDFs
  Screen 2 — Review & Confirm extracted records (editable table)
  Screen 3 — Analysis output (existing pipeline, zero changes)

Tab B (manual entry) is the existing flow — zero modifications.
"""

import logging
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import markdown

from data_models import CultureReport, TrendResult
from extraction import (
    ExtractionError,
    debug_extraction,
    extract_structured_data,
)
from hypothesis import generate_hypothesis
from medgemma import call_medgemma
from pii_removal import detect_pii, scrub_pii
from renderer import render_clinician_output, render_patient_output
from rules import RULES
from trend import analyze_trend

# ---------------------------------------------------------------------------
# Resistance Timeline Renderer
# ---------------------------------------------------------------------------


def render_resistance_timeline(trend: TrendResult) -> str:
    # Defensive: ensure resistance_timeline is List[List[str]]
    timeline = trend.resistance_timeline
    report_dates = trend.report_dates

    # Handle case where data might be serialized/deserialized through Gradio State
    # Gradio may convert lists to Python literal strings (single quotes) not JSON
    if isinstance(timeline, str):
        import ast
        import json

        try:
            # Try JSON first (double quotes)
            timeline = json.loads(timeline)
        except (json.JSONDecodeError, TypeError):
            try:
                # Try Python literal (single quotes)
                timeline = ast.literal_eval(timeline)
            except (ValueError, SyntaxError):
                timeline = []

    if isinstance(report_dates, str):
        import ast
        import json

        try:
            report_dates = json.loads(report_dates)
        except (json.JSONDecodeError, TypeError):
            try:
                report_dates = ast.literal_eval(report_dates)
            except (ValueError, SyntaxError):
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

    header = (
        "| Date | High-Risk Resistance Markers |\n"
        "|------|------------------------------|"
    )
    return header + "\n" + "\n".join(rows)


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
    # Warm white background (main page background)
    body_background_fill="#FDFAF7",
    background_fill_primary="#FDFAF7",
    background_fill_secondary="#F5F0EB",
    # Warm gray borders (#E8DDD6)
    border_color_primary="#E8DDD6",
    block_border_color="#E8DDD6",
    input_border_color="#E8DDD6",
    input_border_color_hover="#E8DDD6",
    # Burnt sienna accent for active elements
    button_primary_background_fill="#C1622F",
    button_primary_background_fill_hover="#a85228",
    button_primary_text_color="#FDFAF7",
    button_secondary_background_fill="#E8DDD6",
    button_secondary_text_color="#5D4037",
    button_cancel_background_fill="#E8DDD6",
    button_cancel_text_color="#5D4037",
    # Form elements with warm gray styling
    checkbox_label_background_fill="#FDFAF7",
    checkbox_label_text_color="#5D4037",
    checkbox_label_text_color_selected="#C1622F",
    checkbox_border_color="#E8DDD6",
    checkbox_border_color_focus="#C1622F",
    # Accordion styling
    accordion_text_color="#C1622F",
    # Subtle shadows only (0 1px 4px with 7% opacity)
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


def _split_manual_reports(text: str) -> List[str]:
    """
    Split manual entry text into separate report blocks.

    Handles multiple formats:
    1. "Report 1", "Report 2" pattern detection
    2. Double newline separator (\\n\\n)
    3. Date-based splitting (multiple "Date:" lines indicate separate reports)
    """
    import re

    if not text or not text.strip():
        return []

    text = text.strip()

    # Try splitting on "Report N" pattern first
    # Match "Report 1", "Report 2", etc. at the start of a line
    report_pattern = re.compile(r'\n(?=Report\s+\d+)', re.IGNORECASE)
    blocks = re.split(report_pattern, text)
    if len(blocks) > 1:
        return [b.strip() for b in blocks if b.strip()]

    # Try splitting on double newline
    blocks = text.split("\n\n")
    if len(blocks) > 1:
        return [b.strip() for b in blocks if b.strip()]

    # Check for multiple "Date:" lines - indicates multiple reports
    date_lines = re.findall(r'^Date:\s*\d{4}-\d{2}-\d{2}', text, re.MULTILINE)
    if len(date_lines) > 1:
        # Split on "Date:" lines, keeping the Date: prefix
        parts = re.split(r'(?=^Date:\s*\d{4}-\d{2}-\d{2})', text, flags=re.MULTILINE)
        return [p.strip() for p in parts if p.strip()]

    # Single block
    return [text] if text else []


def _is_low_confidence(report: CultureReport) -> bool:
    """Return True if any field looks suspiciously generic."""
    return (
        report.organism == "unknown"
        or report.date == "unknown"
        or report.specimen_type not in ("urine", "stool")
        or (
            report.cfu == 0
            and "no growth" not in report.raw_text.lower()
            and report.organism.lower() not in ("no growth",)
            and report.specimen_type != "stool"  # cfu=0 is normal for stool
        )
    )


# ---------------------------------------------------------------------------
# 3. DataFrame helpers (unchanged)
# ---------------------------------------------------------------------------


def _format_susceptibility_summary(report: CultureReport) -> str:
    """Format susceptibility profile as a compact summary string."""
    if not report.susceptibility_profile:
        return "—"

    s_count = sum(1 for s in report.susceptibility_profile if s.interpretation == "S")
    i_count = sum(1 for s in report.susceptibility_profile if s.interpretation == "I")
    r_count = sum(1 for s in report.susceptibility_profile if s.interpretation == "R")

    total = len(report.susceptibility_profile)
    return f"{total} antibiotics: {s_count}S/{i_count}I/{r_count}R"


def reports_to_dataframe_rows(reports: List[CultureReport]) -> List[List[str]]:
    """Convert CultureReport list to list of list strings for gr.Dataframe."""
    rows = []
    for r in reports:
        warn = _WARN_PREFIX if _is_low_confidence(r) else ""
        sus_summary = _format_susceptibility_summary(r)
        rows.append(
            [
                f"{warn}{r.date}",
                r.specimen_type,
                r.organism,
                str(r.cfu),
                ", ".join(r.resistance_markers) if r.resistance_markers else "—",
                sus_summary,
            ]
        )
    return rows


def dataframe_row_to_culture_report(
    row: List[str], original_reports: List[CultureReport] = None
) -> CultureReport:
    """Convert a single Dataframe row (list of strings) back to CultureReport."""
    from rules import normalize_organism

    date_str = row[0].replace(_WARN_PREFIX, "").strip()
    specimen = row[1].strip()
    organism = normalize_organism(row[2].strip())
    cfu_str = row[3].replace(",", "").strip()
    resistance_str = row[4].strip()

    try:
        cfu = int(cfu_str)
    except ValueError:
        cfu = 0

    resistance_markers = (
        [m.strip() for m in resistance_str.split(",") if m.strip() not in ("—", "")]
        if resistance_str != "—"
        else []
    )

    # Try to find matching original report to preserve susceptibility profile
    susceptibility_profile = []
    if original_reports:
        for orig in original_reports:
            # Match by organism (normalized) and CFU value
            orig_organism = normalize_organism(orig.organism)
            if orig_organism == organism and orig.cfu == cfu:
                susceptibility_profile = orig.susceptibility_profile
                break

    return CultureReport(
        date=date_str,
        organism=organism,
        cfu=cfu,
        resistance_markers=resistance_markers,
        susceptibility_profile=susceptibility_profile,  # Preserved from original extraction
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

        # PII/PHI Scrubbing: Remove all patient identifiers before processing
        # First detect what PII is present (for logging/audit)
        pii_detected = detect_pii(markdown_text)
        if pii_detected:
            debug_log += f"  PII detected: {', '.join(pii_detected)}\n"

        # Scrub the PII from the text
        markdown_text = scrub_pii(markdown_text)
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

                # Accept all reports with valid organism/CFU, even if specimen is unknown
                # User can edit specimen type in Review & Confirm screen
                if report.specimen_type not in ("urine", "stool"):
                    debug_log += (
                        f"    ⚠ Specimen type '{report.specimen_type}' detected; "
                        f"user should verify in Review & Confirm\n"
                    )
                else:
                    debug_log += (
                        f"    ✓ Specimen type '{report.specimen_type}' accepted\n"
                    )

                # Override raw_text to the docling markdown block
                report = CultureReport(
                    date=report.date,
                    organism=report.organism,
                    cfu=report.cfu,
                    resistance_markers=report.resistance_markers,
                    susceptibility_profile=report.susceptibility_profile,
                    specimen_type=report.specimen_type,
                    contamination_flag=report.contamination_flag,
                    raw_text=block,  # stored for accordion; never forwarded to MedGemma
                )
                file_reports.append(report)

            except ExtractionError as e:
                debug_log += f"    ✗ ExtractionError: {e}\n"
                pass  # block had no parseable culture data
            except Exception as e:
                debug_log += f"    ✗ Unexpected error: {type(e).__name__}: {e}\n"
                pass

        if not file_reports:
            per_file_statuses.append(
                f'<div style="margin:4px 0"><b>{filename}</b> — '
                f'<span style="color:#e67e22">⚠ No culture data found (check debug output)</span></div>'
            )
            debug_log += f"\n✗ No valid culture records found in {filename}\n\n"
        else:
            # Within-file dedup: if the same PDF was split into multiple blocks that
            # share the same date, prefer the block with a known organism over "unknown".
            # This prevents addendum/section fragments from creating phantom records.
            best_by_date: dict = {}
            best_blocks_by_date: dict = {}
            for r, b in zip(file_reports, [r.raw_text for r in file_reports]):
                existing = best_by_date.get(r.date)
                if existing is None:
                    best_by_date[r.date] = r
                    best_blocks_by_date[r.date] = b
                elif existing.organism == "unknown" and r.organism != "unknown":
                    # Upgrade: replace the unknown block with the informative one
                    debug_log += f"    ✓ Promoted block with organism={r.organism} over unknown for date={r.date}\n"
                    best_by_date[r.date] = r
                    best_blocks_by_date[r.date] = b
                elif r.organism == "unknown":
                    # Phantom fragment: skip in favour of existing
                    debug_log += f"    ⚠ Phantom block (unknown organism, date={r.date}) suppressed\n"
                else:
                    # Different organisms on the same date within one PDF — keep both
                    key = f"{r.date}_{r.organism}"
                    if key not in best_by_date:
                        best_by_date[key] = r
                        best_blocks_by_date[key] = b
            file_reports = list(best_by_date.values())
            file_raw_blocks = list(best_blocks_by_date.values())

            count = len(file_reports)
            per_file_statuses.append(
                f'<div style="margin:4px 0"><b>{filename}</b> — '
                f'<span style="color:#27ae60">✓ {count} record{"s" if count != 1 else ""} found</span></div>'
            )
            all_reports.extend(file_reports)
            all_raw_blocks.extend(file_raw_blocks)
            debug_log += f"\n✓ Extracted {count} record(s) from {filename}\n\n"

    if not all_reports:
        debug_log += "=== RESULT: No valid reports found ===\n"
        return [], [], per_file_statuses, "", debug_log

    # Sort chronologically
    debug_log += f"Sorting {len(all_reports)} report(s) chronologically...\n"
    combined = sorted(zip(all_reports, all_raw_blocks), key=lambda pair: pair[0].date)
    all_reports = [p[0] for p in combined]
    all_raw_blocks = [p[1] for p in combined]

    # TWO-PASS DEDUPLICATION
    # Pass 1: Identify dates that have at least one successful extraction
    dates_with_success: set = set()
    for report in all_reports:
        if report.organism != "unknown" or report.cfu != 0:
            dates_with_success.add(report.date)

    debug_log += f"Dates with successful extractions: {sorted(dates_with_success)}\n"

    # Pass 2: Deduplicate, skipping failed extractions for dates with success
    seen: set = set()
    deduped_reports: List[CultureReport] = []
    deduped_blocks: List[str] = []

    for report, block in zip(all_reports, all_raw_blocks):
        is_failed_extraction = report.organism == "unknown" and report.cfu == 0

        if is_failed_extraction:
            # Skip failed extraction if ANY report for this date has successful extraction
            if report.date in dates_with_success:
                debug_log += f"  ⚠ Failed extraction skipped (successful extraction exists for {report.date})\n"
                continue
            # Also skip if we already have a failed extraction for this date
            key = (report.date, "failed")
            if key in seen:
                debug_log += f"  ⚠ Duplicate failed extraction skipped: date={report.date}\n"
                continue
            seen.add(key)
            deduped_reports.append(report)
            deduped_blocks.append(block)
            debug_log += f"  ⚠ Kept failed extraction (date={report.date})\n"
        else:
            # Successful extraction
            key = (report.date, report.organism, report.cfu)
            if key in seen:
                debug_log += f"  ⚠ Duplicate record skipped: {key}\n"
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
    def run_pipeline(reports: List[CultureReport], progress=None):
        """Run the downstream pipeline with progress tracking."""
        if progress:
            progress(0.1, desc="Sorting reports by date...")

        sorted_reports = sorted(reports, key=lambda r: r.date)

        if progress:
            progress(0.25, desc="Analyzing trends...")
        trend = analyze_trend(sorted_reports)

        if progress:
            progress(0.4, desc="Generating hypothesis...")
        hypothesis = generate_hypothesis(trend, len(sorted_reports))

        if progress:
            progress(0.55, desc="Generating patient explanation...")
        patient_response = call_medgemma(
            trend, hypothesis, "patient", model, tokenizer, is_stub, sorted_reports
        )

        if progress:
            progress(0.75, desc="Generating clinician analysis...")
        clinician_response = call_medgemma(
            trend, hypothesis, "clinician", model, tokenizer, is_stub, sorted_reports
        )

        if progress:
            progress(0.9, desc="Formatting output...")
        patient_out = render_patient_output(
            trend, hypothesis, patient_response, sorted_reports
        )
        clinician_out = render_clinician_output(
            trend, hypothesis, clinician_response, sorted_reports
        )

        if progress:
            progress(1.0, desc="Complete!")

        return trend, patient_out, clinician_out

    def format_output_html(
        patient_out,
        clinician_out,
        trend: TrendResult = None,
        raw_blocks: List[str] = None,
    ) -> Tuple[str, str]:
        """Convert FormattedOutput objects to display HTML — clinical SaaS styling."""
        # ── Patient card ───────────────────────────────────────────────────
        p_body = ""

        # Green improvement alerts for decreasing or cleared trends
        if patient_out.patient_trend_phrase:
            phrase_lower = patient_out.patient_trend_phrase.lower()
            if "downward trend" in phrase_lower:
                # Decreasing trend - improving infection response
                p_body += (
                    "<div class='alert-resolution'>"
                    "<div class='alert-title'>✓ Improving Infection Response</div>"
                    "<div class='alert-text'>"
                    "Declining bacterial counts suggest treatment is working.</div>"
                    "</div>"
                )
            elif "resolution" in phrase_lower:
                # Cleared trend - resolution detected
                p_body += (
                    "<div class='alert-resolution'>"
                    "<div class='alert-title'>✓ Resolution Detected</div>"
                    "<div class='alert-text'>"
                    "Bacterial load has cleared below detection threshold.</div>"
                    "</div>"
                )

        # Info alert for single reports
        if (
            patient_out.patient_trend_phrase
            and "single report" in patient_out.patient_trend_phrase.lower()
        ):
            p_body += (
                "<div style='background:#FDFAF7;border-left:3px solid #D4A574;padding:12px 14px;margin:12px 0;border-radius:6px;'>"
                "<div style='font-size:0.85rem;font-weight:600;color:#7A6558;margin-bottom:4px;'>ℹ Single Report Analysis</div>"
                "<div style='font-size:0.82rem;color:#5D4037;line-height:1.5;'>"
                "This analysis is based on one culture report. For trend analysis (e.g., improving vs worsening infection), "
                "upload 2-3 sequential reports using the <strong>↩ Edit & Re-upload</strong> button.</div>"
                "</div>"
            )

        if patient_out.patient_trend_phrase:
            p_body += (
                f"<p style='font-size:1.0rem;line-height:1.6;margin:0 0 12px 0;'>"
                f"<em>Your results show <strong>{patient_out.patient_trend_phrase}</strong>.</em></p>"
            )
        if patient_out.patient_explanation:
            p_body += (
                f"<div style='line-height:1.6;font-size:0.96rem;'>"
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
                f"<ul style='padding-left:20px;font-size:0.93rem;line-height:1.6;margin:0;'>{qs}</ul>"
            )
        if patient_out.patient_disclaimer:
            p_body += (
                "<div style='margin-top:16px;padding:12px 14px;border:1px solid #E8DDD6;"
                "border-radius:6px;background:#EDE7E0;'>"
                f"<p style='font-size:0.77rem;font-style:italic;color:#5D4037;margin:0;line-height:1.6;'>"
                f"{patient_out.patient_disclaimer}</p></div>"
            )
        patient_html = (
            "<div class='output-card'><h3>📋 Patient Summary</h3>" + p_body + "</div>"
        )

        # ── Clinician card ─────────────────────────────────────────────────
        conf_val = clinician_out.clinician_confidence
        conf_label = f"{conf_val:.0%}" if conf_val is not None else "N/A"

        # Confidence badge (top-right style)
        conf_badge = (
            f"<div class='confidence-badge'>"
            f"<span>Confidence</span><span class='confidence-value'>{conf_label}</span>"
            f"</div>"
        )

        # Header with badge
        c_body = (
            "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;'>"
            "<span></span>" + conf_badge + "</div>"
        )

        # Clinical color logic: #FEE2E2 (red) for stewardship/resistance
        if clinician_out.clinician_stewardship_flag:
            c_body += (
                "<div class='alert-stewardship'>"
                "<div class='alert-title'>⚠ Stewardship Alert</div>"
                "<div class='alert-text'>"
                "Emerging resistance detected — antimicrobial stewardship review recommended.</div>"
                "</div>"
            )

        # Resistance timeline using high-risk markers from data model
        if trend:
            resistance_timeline_str = render_resistance_timeline(trend)
            if resistance_timeline_str != "No high-risk resistance markers detected.":
                # Render table with markers
                c_body += (
                    "<div style='background:#F5F0EB;border-left:3px solid #D4A574;"
                    "padding:12px 14px;margin:12px 0;border-radius:6px;'>"
                    "<p style='margin:0 0 8px;font-size:0.75rem;font-weight:600;text-transform:uppercase;"
                    "letter-spacing:0.04em;color:#7A6558;'>Resistance Timeline</p>"
                    f"<pre style='margin:0;font-size:0.85rem;font-family:monospace;color:#4A3728;"
                    f"white-space:pre-wrap;'>{resistance_timeline_str}</pre></div>"
                )
            else:
                # Show message when no markers exist
                c_body += (
                    "<div style='background:#F5F0EB;border-left:3px solid #D4A574;"
                    "padding:12px 14px;margin:12px 0;border-radius:6px;'>"
                    "<p style='margin:0;font-size:0.85rem;color:#5D4037;'>"
                    "<strong>Resistance Timeline:</strong> No high-risk resistance markers detected.</p></div>"
                )

        if clinician_out.clinician_interpretation:
            # Convert markdown to HTML for proper rendering (bold, tables, etc.)
            html_content = markdown.markdown(
                clinician_out.clinician_interpretation,
                extensions=['tables', 'fenced_code']
            )
            c_body += (
                f"<div style='line-height:1.6;font-size:0.96rem;margin-top:12px;'>"
                f"{html_content}</div>"
            )
        if clinician_out.clinician_disclaimer:
            c_body += (
                "<p style='font-style:italic;color:#7A6558;border-top:1px solid #E8DDD6;"
                "padding-top:12px;margin-top:18px;font-size:0.77rem;line-height:1.6;'>"
                f"{clinician_out.clinician_disclaimer}</p>"
            )

        # Raw extracted text accordion (if provided)
        if raw_blocks:
            raw_sections = ""
            for i, block in enumerate(raw_blocks, 1):
                raw_sections += (
                    f"<div style='margin-bottom:12px;'>"
                    f"<p style='font-size:0.7rem;font-weight:600;color:#7A6558;margin:0 0 4px;'>"
                    f"Record {i}</p>"
                    f"<pre style='margin:0;padding:10px;background:#F5F0EB;border:1px solid #E8DDD6;"
                    f"border-radius:4px;font-size:0.8rem;overflow-x:auto;'>{block}</pre></div>"
                )
            c_body += (
                "<div style='margin-top:16px;border:1px solid #E8DDD6;border-radius:6px;overflow:hidden;'>"
                "<details style='background:#FDFAF7;'>"
                "<summary style='padding:12px 14px;font-size:0.8rem;font-weight:600;color:#5D4037;"
                "cursor:pointer;background:#F5F0EB;border-bottom:1px solid #E8DDD6;'>"
                "📋 View Source Data</summary>"
                f"<div style='padding:14px;'>{raw_sections}</div>"
                "</details></div>"
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

        /* Container - prevent squished text with max-width and centering */
        .gradio-container {
            max-width: 1150px !important;
            margin: 0 auto !important;
            padding: 40px !important;
        }

        /* Main content wrapper for better readability */
        .container {
            max-width: 1100px !important;
            margin: 0 auto !important;
        }

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

        /* Output cards - medical journal style with improved readability */
        .output-card {
            border: 1px solid #E8DDD6;
            border-radius: 4px;
            padding: 22px 26px;
            background: #FDFAF7;
            margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(28,20,18,0.07);
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 0.96rem;
            line-height: 1.6;
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

        /* Clinical alert boxes - traffic light system */
        .alert-stewardship {
            background: #FEE2E2 !important;
            border-left: 3px solid #DC2626 !important;
            padding: 12px 14px;
            margin: 12px 0;
            border-radius: 6px;
        }
        .alert-stewardship .alert-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #991B1B;
        }
        .alert-stewardship .alert-text {
            margin: 4px 0 0;
            font-size: 0.82rem;
            color: #7F1D1D;
            line-height: 1.5;
        }

        .alert-resolution {
            background: #DCFCE7 !important;
            border-left: 3px solid #16A34A !important;
            padding: 12px 14px;
            margin: 12px 0;
            border-radius: 6px;
        }
        .alert-resolution .alert-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #166534;
        }
        .alert-resolution .alert-text {
            margin: 4px 0 0;
            font-size: 0.82rem;
            color: #14532D;
            line-height: 1.5;
        }

        /* Confidence badge - compact top-right style */
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: #F5F0EB;
            border: 1px solid #E8DDD6;
            border-radius: 12px;
            font-family: system-ui, sans-serif;
            font-size: 0.75rem;
            font-weight: 600;
            color: #7A6558;
        }
        .confidence-badge .confidence-value {
            color: #2563EB;
            font-size: 0.85rem;
        }

        /* Resistance timeline table */
        .resistance-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 0.85rem;
            margin: 8px 0;
        }
        .resistance-table th {
            background: #F5F0EB;
            border: 1px solid #E8DDD6;
            padding: 8px 10px;
            text-align: left;
            font-family: system-ui, sans-serif;
            font-size: 0.7rem;
            font-weight: 600;
            color: #7A6558;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .resistance-table td {
            border: 1px solid #E8DDD6;
            padding: 8px 10px;
            color: #4A3728;
        }
        .resistance-table tr:nth-child(even) {
            background: #FAF7F4;
        }
        .resistance-table .marker-s { color: #16A34A; font-weight: 600; }
        .resistance-table .marker-i { color: #D97706; font-weight: 600; }
        .resistance-table .marker-r { color: #DC2626; font-weight: 600; }

        /* Hypotheses table - handle longer evidence text */
        .output-card table {
            width: 100%;
            border-collapse: collapse;
        }
        .output-card table td {
            white-space: normal;
            word-wrap: break-word;
            max-width: 300px;
            vertical-align: top;
        }

        /* Scrollable textbox for raw extracted text */
        .raw-textbox textarea {
            min-height: 200px;
            max-height: 500px;
            overflow-y: auto !important;
            white-space: pre-wrap;
            word-wrap: break-word;
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

        /* Input fields - warm classical styling */
        .gr-input {
            border: 1px solid #E8DDD6;
            border-radius: 4px;
            background: #FDFAF7;
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 0.9rem;
            color: #4A3728;
        }
        .gr-input:focus {
            outline: none;
            border-color: #C1622F;
            box-shadow: 0 0 0 2px rgba(193, 98, 47, 0.1);
        }

        /* Buttons - distinct primary action styling */
        .gr-button {
            font-family: system-ui, sans-serif !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.04em !important;
            text-transform: uppercase !important;
            border: 1px solid #E8DDD6 !important;
            border-radius: 4px !important;
            transition: all 0.2s ease !important;
        }
        .gr-button:hover {
            border-color: #2563EB !important;
        }
        .gr-button.primary {
            background: #2563EB !important;
            border-color: #2563EB !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
        }
        .gr-button.primary:hover {
            background: #1D4ED8 !important;
            border-color: #1D4ED8 !important;
            box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3) !important;
        }

        /* Tabs - subtle warm styling */
        .tabitem {
            border: 1px solid #E8DDD6;
            border-radius: 4px;
            background: #FDFAF7;
            box-shadow: 0 1px 4px rgba(28,20,18,0.07);
        }
        .tab-nav {
            border-bottom: 1px solid #E8DDD6;
            background: #F5F0EB;
        }
        .tab-nav button {
            font-family: system-ui, sans-serif !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.04em !important;
            text-transform: uppercase !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            color: #7A6558 !important;
            padding: 10px 16px !important;
        }
        .tab-nav button:hover {
            color: #2563EB !important;
            background: rgba(37, 99, 235, 0.05);
        }
        .tab-nav button.selected {
            border-bottom-color: #2563EB !important;
            color: #2563EB !important;
        }

        /* Dataframe - table styling */
        .dataframe {
            border: 1px solid #E8DDD6;
            border-radius: 4px;
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 0.85rem;
        }
        .dataframe th {
            background: #F5F0EB;
            border-bottom: 1px solid #E8DDD6;
            font-family: system-ui, sans-serif;
            font-size: 0.75rem;
            font-weight: 600;
            color: #7A6558;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .dataframe td {
            border-bottom: 1px solid #E8DDD6;
            color: #4A3728;
        }
        .dataframe input {
            border: 1px solid #E8DDD6;
            border-radius: 2px;
            background: #FDFAF7;
            font-family: 'Source Serif 4', Georgia, serif;
        }

        /* Accordion - for Raw Extracted Text */
        .accordion {
            border: 1px solid #E8DDD6;
            border-radius: 4px;
            background: #FDFAF7;
        }
        .accordion-header {
            font-family: system-ui, sans-serif !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            color: #5D4037 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }
        .accordion-header:hover {
            color: #2563EB !important;
        }

        /* Status Indicator Panel */
        .status-panel-container {
            background: #F5F0EB !important;
            border: 1px solid #E8DDD6 !important;
            border-radius: 6px !important;
            padding: 12px 20px 12px 24px !important;
            font-family: system-ui, sans-serif !important;
            font-size: 0.82rem !important;
            margin-bottom: 16px !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            gap: 32px !important;
            overflow: visible !important;
        }
        #pii_status, #medgemma_status {
            font-family: system-ui, sans-serif !important;
            font-size: 0.82rem !important;
            margin: 0 0 0 8px !important;
            padding: 0 !important;
        }
        #pii_status .status-light, #medgemma_status .status-light {
            margin-left: 4px !important;
        }
        /* Status Light Indicators */
        .status-light {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
            flex-shrink: 0;
        }
        .status-light-white {
            background: #D1D5DB;
            border: 1px solid #9CA3AF;
        }
        .status-light-green {
            background: #22C55E;
            box-shadow: 0 0 6px 2px rgba(34, 197, 94, 0.5);
        }
        .status-light-blue {
            background: #3B82F6;
            box-shadow: 0 0 6px 2px rgba(59, 130, 246, 0.5);
        }
        /* Ensure no clipping on status panel */
        .status-panel-container > div {
            overflow: visible !important;
        }
    """,
    ) as demo:
        gr.Markdown(
            "# 🧫 CultureSense — Longitudinal Clinical Hypothesis Engine\n\n*Powered by MedGemma 4B-IT*"
        )
        gr.Markdown(
            "**Upload 2–3 sequential urine or stool culture reports** to analyze trends over time and generate a clinical hypothesis. "
            "While the pipeline is designed for longitudinal analysis, single reports are also supported to help you understand your culture results."
        )

        # ── Pipeline Status Indicators ──────────────────────────────────────────────
        with gr.Row(
            visible=False, elem_classes="status-panel-container"
        ) as status_indicator_panel:
            pii_status = gr.Markdown(
                value='<span class="status-light status-light-white"></span>Awaiting upload...',
                elem_id="pii_status",
            )
            medgemma_status = gr.Markdown(
                value='<span class="status-light status-light-white"></span>Awaiting analysis...',
                elem_id="medgemma_status",
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
                            "Susceptibility Profile",
                        ],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        interactive=True,
                        wrap=True,
                        label="Extracted Culture Records",
                    )

                    with gr.Accordion(
                        "📋 Raw Extracted Text (for clinician verification)",
                        open=False,
                    ):
                        raw_box_0 = gr.Textbox(
                            label="Record 1",
                            interactive=False,
                            visible=False,
                            container=True,
                            show_label=True,
                            elem_classes="raw-textbox",
                        )
                        raw_box_1 = gr.Textbox(
                            label="Record 2",
                            interactive=False,
                            visible=False,
                            container=True,
                            show_label=True,
                            elem_classes="raw-textbox",
                        )
                        raw_box_2 = gr.Textbox(
                            label="Record 3",
                            interactive=False,
                            visible=False,
                            container=True,
                            show_label=True,
                            elem_classes="raw-textbox",
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
                    output_patient_md = gr.Markdown(value="")
                    output_clinician_md = gr.Markdown(value="")
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
                            gr.update(visible=False),  # btn_fullscreen_0
                            gr.update(visible=False),  # btn_fullscreen_1
                            gr.update(visible=False),  # btn_fullscreen_2
                            "",  # debug_output
                            gr.update(visible=True),  # btn_process
                            gr.update(visible=False),  # btn_process_loading
                            gr.update(visible=False),  # loading_html
                            gr.update(visible=False),  # status_indicator_panel
                            '<span class="status-light status-light-white"></span>Awaiting upload...',  # pii_status
                            '<span class="status-light status-light-white"></span>Awaiting analysis...',  # medgemma_status
                        )

                    reports, raw_blocks, statuses, trunc_warn, debug_log = (
                        process_uploaded_pdfs(files)
                    )
                    # Add header showing total PDFs uploaded
                    pdf_count = len(files) if files else 0
                    status_header = (
                        f'<div style="margin-bottom:8px;padding:8px 12px;background:#f0f0f0;'
                        f'border-radius:4px;font-weight:500;">'
                        f"📄 {pdf_count} PDF{'s' if pdf_count != 1 else ''} uploaded</div>"
                    )
                    status_combined = status_header + "".join(statuses)

                    if not reports:
                        # All files failed — stay on screen 1, show error panel
                        error_msg = (
                            status_header
                            + '<div style="padding:12px;background:#f8d7da;border:1px solid #f5c6cb;border-radius:4px;color:#721c24;">'
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
                            gr.update(visible=False),  # status_indicator_panel
                            '<span class="status-light status-light-white"></span>Awaiting upload...',  # pii_status
                            '<span class="status-light status-light-white"></span>Awaiting analysis...',  # medgemma_status
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
                        gr.update(visible=True),  # status_indicator_panel
                        '<span class="status-light status-light-green"></span>PII/PHI removed — all patient identifiers redacted',  # pii_status
                        '<span class="status-light status-light-white"></span>Awaiting analysis...',  # medgemma_status
                    )

                # Chain the events: first show loading, then process
                # NOTE: confirm_table is ONLY updated in on_process_pdfs, not in
                # on_process_pdfs_start, to prevent duplicate rendering
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
                        status_indicator_panel,
                        pii_status,
                        medgemma_status,
                    ],
                )

                # ── Event: Confirm & Analyse ────────────────────────────────
                def on_confirm_start():
                    """Show analyzing status immediately when confirm button is clicked."""
                    return '<span class="status-light status-light-blue"></span>MedGemma analyzing...'

                def on_confirm(
                    table_data, raw_blocks, original_reports, progress=gr.Progress()
                ):
                    if table_data is None or len(table_data) == 0:
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            "<p style='color:#c0392b'>No records to analyse.</p>",
                            "",
                            '<span class="status-light status-light-white"></span>Awaiting analysis...',  # medgemma_status (no change)
                        )

                    # Handle different DataFrame formats from Gradio
                    # table_data can be: pandas DataFrame, list of lists, or numpy array
                    rows = []
                    try:
                        import pandas as pd

                        if isinstance(table_data, pd.DataFrame):
                            # Convert DataFrame to list of lists (values only, no headers)
                            rows = table_data.values.tolist()
                        elif hasattr(table_data, "tolist"):
                            # numpy array or similar
                            rows = table_data.tolist()
                        elif isinstance(table_data, (list, tuple)):
                            rows = list(table_data)
                        else:
                            rows = []
                    except Exception as e:
                        logging.warning(
                            f"DEBUG on_confirm: error converting table_data: {e}"
                        )
                        rows = []

                    # Filter out header rows and invalid data
                    # Headers are typically: ["Date", "Specimen", "Organism", "CFU/mL", "Resistance Markers"]
                    header_indicators = [
                        "Date",
                        "date",
                        "Specimen",
                        "Organism",
                        "CFU",
                        "Resistance",
                    ]
                    data_rows = []
                    for row in rows:
                        # Skip if row is not a list/tuple
                        if not isinstance(row, (list, tuple)) or len(row) < 5:
                            continue
                        # Skip header rows - check if first cell contains header text
                        first_cell = str(row[0]) if row[0] is not None else ""
                        if any(
                            indicator in first_cell for indicator in header_indicators
                        ):
                            continue
                        data_rows.append(row)

                    # Convert edited table rows back to CultureReport objects
                    confirmed_reports = []
                    for row in data_rows:
                        try:
                            report = dataframe_row_to_culture_report(
                                row, original_reports
                            )
                            confirmed_reports.append(report)
                        except Exception:
                            pass

                    if not confirmed_reports:
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            "<p style='color:#c0392b'>Could not parse records.</p>",
                            "",
                            '<span class="status-light status-light-white"></span>Awaiting analysis...',  # medgemma_status (no change)
                        )

                    try:
                        trend, patient_out, clinician_out = run_pipeline(
                            confirmed_reports, progress
                        )
                        patient_html, clinician_html = format_output_html(
                            patient_out, clinician_out, trend, raw_blocks
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
                        '<span class="status-light status-light-blue"></span>Analysis complete',  # medgemma_status
                    )

                # Chain the events: first show analyzing status, then run pipeline
                btn_confirm.click(
                    fn=on_confirm_start,
                    inputs=[],
                    outputs=[medgemma_status],
                ).then(
                    fn=on_confirm,
                    inputs=[confirm_table, state_raw_blocks, state_reports],
                    outputs=[
                        screen_confirm,
                        screen_upload,
                        screen_output,
                        output_patient_md,
                        output_clinician_md,
                        medgemma_status,
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
                        gr.update(visible=False),  # hide status_indicator_panel
                        '<span class="status-light status-light-white"></span>Awaiting upload...',  # reset pii_status
                        '<span class="status-light status-light-white"></span>Awaiting analysis...',  # reset medgemma_status
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
                        status_indicator_panel,
                        pii_status,
                        medgemma_status,
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
            # TAB B — Manual Entry (updated with status indicators)
            # ================================================================
            with gr.Tab("✏ Enter Manually", id="tab_manual"):
                gr.Markdown("### Paste culture report text directly")
                gr.Markdown(
                    "Paste 2–3 sequential culture reports. "
                    "The pipeline will extract structured data, analyse trends, and generate hypotheses."
                )

                # Status indicator panel for Manual tab
                with gr.Row(visible=True, elem_classes="status-panel-container") as status_indicator_panel_manual:
                    pii_status_manual = gr.Markdown(
                        value='<span class="status-light status-light-white"></span>Ready...',
                        elem_id="pii_status_manual",
                    )
                    medgemma_status_manual = gr.Markdown(
                        value='<span class="status-light status-light-white"></span>Awaiting analysis...',
                        elem_id="medgemma_status_manual",
                    )

                manual_input = gr.Textbox(
                    label="Culture Reports (2–3 sequential)",
                    placeholder="Paste report text here...\n\nExample format:\nReport 1\nDate: 2024-01-15\nOrganism: E. coli\nCFU: 100000\n...\n\nReport 2\nDate: 2024-01-22\nOrganism: E. coli\nCFU: 50000\n...",
                    lines=12,
                )
                btn_analyse_manual = gr.Button("🔬 Analyse", variant="primary")
                manual_output_patient = gr.Markdown()
                manual_output_clinician = gr.Markdown()

                def on_analyse_manual_start():
                    """Show analyzing status immediately when button is clicked."""
                    return (
                        '<span class="status-light status-light-green"></span>No PII detected (manual entry)',
                        '<span class="status-light status-light-blue"></span>MedGemma analyzing...',
                    )

                def on_analyse_manual(text, progress=gr.Progress()):
                    if not text or len(text.strip()) < 20:
                        return (
                            "<p style='color:#c0392b'>Please paste at least one full report.</p>",
                            "",
                            '<span class="status-light status-light-white"></span>Ready...',
                            '<span class="status-light status-light-white"></span>Awaiting analysis...',
                        )

                    # Split using the new smart splitter
                    blocks = _split_manual_reports(text)

                    reports = []
                    for block in blocks:
                        try:
                            # Scrub PII first (defense in depth)
                            clean_block = scrub_pii(block)
                            r = extract_structured_data(clean_block)
                            reports.append(r)
                        except Exception:
                            pass

                    if len(reports) < 1:
                        return (
                            "<p style='color:#c0392b'>Could not extract data from pasted text. "
                            "Check format includes Date, Organism, and CFU/mL.</p>",
                            "",
                            '<span class="status-light status-light-green"></span>No PII detected (manual entry)',
                            '<span class="status-light status-light-white"></span>Awaiting analysis...',
                        )

                    try:
                        trend, patient_out, clinician_out = run_pipeline(reports, progress)
                        patient_html, clinician_html = format_output_html(
                            patient_out, clinician_out, trend
                        )
                    except Exception as e:
                        patient_html = (
                            f"<p style='color:#c0392b'>Analysis error: {e}</p>"
                        )
                        clinician_html = ""

                    return (
                        patient_html,
                        clinician_html,
                        '<span class="status-light status-light-green"></span>No PII detected (manual entry)',
                        '<span class="status-light status-light-blue"></span>Analysis complete',
                    )

                # Chain the events: first show status, then run analysis
                btn_analyse_manual.click(
                    fn=on_analyse_manual_start,
                    inputs=[],
                    outputs=[pii_status_manual, medgemma_status_manual],
                ).then(
                    fn=on_analyse_manual,
                    inputs=[manual_input],
                    outputs=[manual_output_patient, manual_output_clinician, pii_status_manual, medgemma_status_manual],
                )

    return demo
