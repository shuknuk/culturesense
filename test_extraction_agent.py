"""
Unit tests for CultureSense Extraction Agent.

Tests cover:
- _split_into_report_blocks: separator-based splitting
- reports_to_dataframe_rows: low-confidence prefix, resistance formatting
- dataframe_row_to_culture_report: raw_text="", warning prefix stripping, resistance parsing
- process_uploaded_pdfs (mocked): dedup, sort, truncation, all-fail path
"""

import warnings
from unittest.mock import MagicMock, patch
from dataclasses import replace

import pytest

# Import the module under test
from extraction_agent import (
    _split_into_report_blocks,
    reports_to_dataframe_rows,
    dataframe_row_to_culture_report,
    process_uploaded_pdfs,
    _is_low_confidence,
    MAX_RECORDS,
    _WARN_PREFIX,
)
from data_models import CultureReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_report(date="2026-01-01", organism="Escherichia coli", cfu=120000,
                resistance=None, specimen="urine", contamination=False):
    return CultureReport(
        date=date,
        organism=organism,
        cfu=cfu,
        resistance_markers=resistance or [],
        specimen_type=specimen,
        contamination_flag=contamination,
        raw_text="<raw>",
    )


# ---------------------------------------------------------------------------
# _split_into_report_blocks
# ---------------------------------------------------------------------------

class TestSplitIntoReportBlocks:
    def test_single_block_no_separator(self):
        text = "Date: 2026-01-01\nOrganism: E. coli\nCFU/mL: 120000"
        blocks = _split_into_report_blocks(text)
        assert len(blocks) == 1
        assert "E. coli" in blocks[0]

    def test_splits_on_horizontal_rule(self):
        text = "Report 1\nOrganism: E. coli\n\n---\n\nReport 2\nOrganism: Klebsiella"
        blocks = _split_into_report_blocks(text)
        assert len(blocks) == 2
        assert "E. coli" in blocks[0]
        assert "Klebsiella" in blocks[1]

    def test_splits_on_heading(self):
        text = "# Report 1\nOrganism: E. coli\n# Report 2\nOrganism: Klebsiella"
        blocks = _split_into_report_blocks(text)
        assert len(blocks) == 2

    def test_empty_text_returns_empty(self):
        blocks = _split_into_report_blocks("")
        assert blocks == []

    def test_whitespace_only_returns_empty(self):
        blocks = _split_into_report_blocks("   \n  ")
        assert blocks == []


# ---------------------------------------------------------------------------
# _is_low_confidence
# ---------------------------------------------------------------------------

class TestIsLowConfidence:
    def test_unknown_organism_is_low(self):
        r = make_report(organism="unknown")
        assert _is_low_confidence(r) is True

    def test_zero_cfu_known_organism_is_low(self):
        r = make_report(cfu=0)
        assert _is_low_confidence(r) is True

    def test_normal_report_is_not_low(self):
        r = make_report(organism="Escherichia coli", cfu=120000)
        assert _is_low_confidence(r) is False


# ---------------------------------------------------------------------------
# reports_to_dataframe_rows
# ---------------------------------------------------------------------------

class TestReportsToDataframeRows:
    def test_basic_row_structure(self):
        r = make_report(resistance=["ESBL", "CRE"])
        rows = reports_to_dataframe_rows([r])
        assert len(rows) == 1
        row = rows[0]
        assert row[0] == "2026-01-01"   # Date
        assert row[1] == "urine"         # Specimen
        assert row[2] == "Escherichia coli"  # Organism (no prefix)
        assert row[3] == "120000"        # CFU/mL
        assert "ESBL" in row[4]
        assert "CRE" in row[4]

    def test_no_resistance_shows_none(self):
        r = make_report(resistance=[])
        rows = reports_to_dataframe_rows([r])
        assert rows[0][4] == "None"

    def test_low_confidence_adds_prefix(self):
        r = make_report(organism="unknown")
        rows = reports_to_dataframe_rows([r])
        assert rows[0][2].startswith(_WARN_PREFIX)

    def test_multiple_reports(self):
        reports = [make_report(date=f"2026-01-0{i}") for i in range(1, 4)]
        rows = reports_to_dataframe_rows(reports)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# dataframe_row_to_culture_report
# ---------------------------------------------------------------------------

class TestDataframeRowToCultureReport:
    def _make_row(self, date="2026-01-01", specimen="urine",
                  organism="Escherichia coli", cfu="120000",
                  resistance="ESBL, CRE"):
        return [date, specimen, organism, cfu, resistance]

    def test_raw_text_is_always_empty(self):
        row = self._make_row()
        report = dataframe_row_to_culture_report(row)
        assert report.raw_text == ""

    def test_warn_prefix_stripped_from_organism(self):
        row = self._make_row(organism=f"{_WARN_PREFIX}unknown")
        report = dataframe_row_to_culture_report(row)
        assert report.organism == "unknown"
        assert _WARN_PREFIX not in report.organism

    def test_resistance_parsed_correctly(self):
        row = self._make_row(resistance="ESBL, CRE, MRSA")
        report = dataframe_row_to_culture_report(row)
        assert report.resistance_markers == ["ESBL", "CRE", "MRSA"]

    def test_none_resistance_gives_empty_list(self):
        row = self._make_row(resistance="None")
        report = dataframe_row_to_culture_report(row)
        assert report.resistance_markers == []

    def test_cfu_with_commas_parsed(self):
        row = self._make_row(cfu="120,000")
        report = dataframe_row_to_culture_report(row)
        assert report.cfu == 120000

    def test_invalid_cfu_defaults_to_zero(self):
        row = self._make_row(cfu="not_a_number")
        report = dataframe_row_to_culture_report(row)
        assert report.cfu == 0

    def test_contamination_flag_set(self):
        row = self._make_row(organism="mixed flora")
        report = dataframe_row_to_culture_report(row)
        assert report.contamination_flag is True

    def test_contamination_flag_not_set_for_normal(self):
        row = self._make_row(organism="Escherichia coli")
        report = dataframe_row_to_culture_report(row)
        assert report.contamination_flag is False


# ---------------------------------------------------------------------------
# process_uploaded_pdfs (mocked Docling + extract_structured_data)
# ---------------------------------------------------------------------------

class TestProcessUploadedPdfs:
    """Tests with mocked Docling and extract_structured_data."""

    def _make_file(self, name="report.pdf"):
        f = MagicMock()
        f.name = f"/tmp/{name}"
        return f

    def _make_report(self, date, specimen="urine"):
        return CultureReport(
            date=date,
            organism="Escherichia coli",
            cfu=120000,
            resistance_markers=[],
            specimen_type=specimen,
            contamination_flag=False,
            raw_text="<raw>",
        )

    @patch("extraction_agent.process_pdf_file")
    @patch("extraction_agent.extract_structured_data")
    def test_single_file_single_report(self, mock_extract, mock_pdf):
        mock_pdf.return_value = ("Date: 2026-01-01\nOrganism: E. coli\nCFU/mL: 120000\nSpecimen: urine", "")
        mock_extract.return_value = self._make_report("2026-01-01")

        files = [self._make_file("report.pdf")]
        reports, blocks, statuses, trunc = process_uploaded_pdfs(files)

        assert len(reports) == 1
        assert reports[0].date == "2026-01-01"
        assert trunc == ""
        assert "✓" in statuses[0]

    @patch("extraction_agent.process_pdf_file")
    @patch("extraction_agent.extract_structured_data")
    def test_three_files_merged_and_sorted(self, mock_extract, mock_pdf):
        mock_pdf.return_value = ("Date: 2026-01-01\nOrganism: E. coli\nCFU/mL: 120000\nSpecimen: urine", "")
        mock_extract.side_effect = [
            self._make_report("2026-01-20"),
            self._make_report("2026-01-01"),
            self._make_report("2026-01-10"),
        ]

        files = [self._make_file(f"r{i}.pdf") for i in range(3)]
        reports, _, _, _ = process_uploaded_pdfs(files)

        dates = [r.date for r in reports]
        assert dates == sorted(dates), "Reports should be sorted chronologically"

    @patch("extraction_agent.process_pdf_file")
    def test_all_files_fail_returns_empty(self, mock_pdf):
        mock_pdf.return_value = ("", '<span style="color:#c0392b">✗ Could not read</span>')

        files = [self._make_file("bad.pdf")]
        reports, blocks, statuses, trunc = process_uploaded_pdfs(files)

        assert reports == []
        assert blocks == []
        assert "✗" in statuses[0]

    @patch("extraction_agent.process_pdf_file")
    @patch("extraction_agent.extract_structured_data")
    def test_deduplication(self, mock_extract, mock_pdf):
        mock_pdf.return_value = ("text", "")
        # Same date+organism+cfu twice
        dup = self._make_report("2026-01-01")
        mock_extract.side_effect = [dup, dup]

        files = [self._make_file(f"r{i}.pdf") for i in range(2)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reports, _, _, _ = process_uploaded_pdfs(files)
            assert len(reports) == 1
            assert any("Duplicate" in str(warning.message) for warning in w)

    @patch("extraction_agent.process_pdf_file")
    @patch("extraction_agent.extract_structured_data")
    def test_truncation_to_max_records(self, mock_extract, mock_pdf):
        mock_pdf.return_value = ("text", "")
        mock_extract.side_effect = [
            self._make_report(f"2026-01-0{i}") for i in range(1, 5)
        ]

        files = [self._make_file(f"r{i}.pdf") for i in range(4)]
        reports, _, _, trunc_warn = process_uploaded_pdfs(files)

        assert len(reports) == MAX_RECORDS
        assert trunc_warn != ""  # Warning should be present
        # Should keep most recent
        assert reports[-1].date == "2026-01-04"

    @patch("extraction_agent.process_pdf_file")
    @patch("extraction_agent.extract_structured_data")
    def test_non_urine_stool_filtered_out(self, mock_extract, mock_pdf):
        mock_pdf.return_value = ("text", "")
        blood_report = CultureReport(
            date="2026-01-01", organism="Staph", cfu=5000,
            resistance_markers=[], specimen_type="blood",
            contamination_flag=False, raw_text=""
        )
        mock_extract.return_value = blood_report

        files = [self._make_file("blood.pdf")]
        reports, _, statuses, _ = process_uploaded_pdfs(files)

        assert reports == []
        assert "⚠" in statuses[0]  # amber warning for no culture data

    @patch("extraction_agent.process_pdf_file")
    @patch("extraction_agent.extract_structured_data")
    def test_one_bad_one_good_file(self, mock_extract, mock_pdf):
        def pdf_side_effect(path):
            if "bad" in path:
                return ("", '<span style="color:#c0392b">✗ Could not read</span>')
            return ("text", "")

        mock_pdf.side_effect = pdf_side_effect
        mock_extract.return_value = self._make_report("2026-01-01")

        files = [self._make_file("bad.pdf"), self._make_file("good.pdf")]
        reports, _, statuses, _ = process_uploaded_pdfs(files)

        assert len(reports) == 1
        assert "✗" in statuses[0]   # bad file flagged
        assert "✓" in statuses[1]   # good file processed

    def test_empty_file_list_returns_empty(self):
        reports, blocks, statuses, trunc = process_uploaded_pdfs([])
        assert reports == []
        assert blocks == []
        assert statuses == []
        assert trunc == ""
