"""
CultureSense Extraction Layer
Parses free-text culture reports into typed CultureReport dataclasses.
"""

import re
import warnings
from typing import Optional

from data_models import CultureReport
from rules import RULES, normalize_organism


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class ExtractionError(ValueError):
    """Raised when both organism AND cfu fail to parse from a report."""


# ---------------------------------------------------------------------------
# Compiled regex patterns (Section 5.2)
# ---------------------------------------------------------------------------

# Organism: "Organism: <value>" up to newline or end of string
_RE_ORGANISM = re.compile(r"Organism:\s*(.+?)(?:\n|$)", re.IGNORECASE)

# CFU/mL: "CFU/mL: <digits with optional commas>"
_RE_CFU_PRIMARY = re.compile(r"CFU/mL:\s*([\d,]+)", re.IGNORECASE)

# Fallback CFU patterns
_RE_CFU_SCIENTIFIC = re.compile(r"10\^(\d+)", re.IGNORECASE)  # 10^5 → 100000
_RE_CFU_WORD = re.compile(r"(TNTC|Too\s+Numerous\s+To\s+Count)", re.IGNORECASE)
_RE_CFU_NO_GROWTH = re.compile(r"(No\s+growth|0\s+CFU)", re.IGNORECASE)
_RE_CFU_RAW_NUMBER = re.compile(r"\b([\d]{4,})\b")  # bare large number

# Date: ISO 8601 or MM/DD/YYYY
_RE_DATE_PRIMARY = re.compile(
    r"(?:Date|Collected|Reported):\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)
_RE_DATE_FALLBACK = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Resistance markers: exact case-insensitive word boundaries
_RE_RESISTANCE = re.compile(r"\b(ESBL|CRE|MRSA|VRE|CRKP)\b", re.IGNORECASE)

# Specimen type
_RE_SPECIMEN = re.compile(r"Specimen:\s*(urine|stool|wound|blood)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# CFU normalisation helper (Section 5.4)
# ---------------------------------------------------------------------------


def _parse_cfu(report_text: str) -> tuple[int, bool]:
    """
    Attempt to parse the CFU/mL value from a report text string.

    Returns:
        (cfu_value, parse_success) tuple.

    Normalisation rules:
        - "TNTC" / "Too Numerous To Count" → 999999
        - "No growth" / "0 CFU"            → 0
        - "10^5"                            → 100000
        - comma-separated integer           → int (commas stripped)
        - Missing/unparseable               → 0 with warning
    """
    # 1. Primary: "CFU/mL: 120,000"
    m = _RE_CFU_PRIMARY.search(report_text)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            return int(raw), True
        except ValueError:
            pass

    # 2. TNTC
    if _RE_CFU_WORD.search(report_text):
        return 999999, True

    # 3. No growth
    if _RE_CFU_NO_GROWTH.search(report_text):
        return 0, True

    # 4. Scientific notation "10^5"
    m = _RE_CFU_SCIENTIFIC.search(report_text)
    if m:
        try:
            return 10 ** int(m.group(1)), True
        except (ValueError, OverflowError):
            pass

    # 5. Bare large integer (≥4 digits) — last resort fallback
    m = _RE_CFU_RAW_NUMBER.search(report_text)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            val = int(raw)
            warnings.warn(
                f"CFU parsed from bare number '{raw}' — review report text.",
                UserWarning,
                stacklevel=3,
            )
            return val, True
        except ValueError:
            pass

    warnings.warn(
        "CFU/mL could not be parsed; defaulting to 0.", UserWarning, stacklevel=3
    )
    return 0, False


def _parse_date(report_text: str) -> str:
    """Extract and normalise the collection date from report text."""
    m = _RE_DATE_PRIMARY.search(report_text)
    if m:
        raw = m.group(1)
        # Convert MM/DD/YYYY → YYYY-MM-DD
        if "/" in raw:
            parts = raw.split("/")
            return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
        return raw

    # Fallback: any ISO date in text
    m = _RE_DATE_FALLBACK.search(report_text)
    if m:
        return m.group(1)

    return "unknown"


def _parse_organism(report_text: str) -> Optional[str]:
    """
    Extract organism name from report text.

    Primary: "Organism: <value>"
    Fallback: scan for known organism names / aliases.
    """
    m = _RE_ORGANISM.search(report_text)
    if m:
        return normalize_organism(m.group(1).strip())

    # Fallback: search for known organism aliases in full text
    lower_text = report_text.lower()
    from rules import ORGANISM_ALIASES

    for alias in sorted(ORGANISM_ALIASES.keys(), key=len, reverse=True):
        if alias in lower_text:
            return normalize_organism(alias)

    return None


def _parse_resistance_markers(report_text: str) -> list[str]:
    """Extract all high-risk resistance markers (deduplicated, uppercase)."""
    found = _RE_RESISTANCE.findall(report_text)
    return list(dict.fromkeys(m.upper() for m in found))  # deduplicate, preserve order


def _parse_specimen(report_text: str) -> str:
    """Extract specimen type; defaults to 'unknown'."""
    m = _RE_SPECIMEN.search(report_text)
    if m:
        return m.group(1).lower()
    return "unknown"


def _is_contamination(organism: str) -> bool:
    """Return True if the organism name matches any contamination term."""
    lower = organism.lower()
    return any(term in lower for term in RULES["contamination_terms"])


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------


def extract_structured_data(report_text: str) -> CultureReport:
    """
    Parse a free-text culture report into a typed CultureReport.

    Rules:
        - Organism field: stripped, normalised via ORGANISM_ALIASES
        - CFU: commas removed, converted to int; TNTC=999999
        - resistance_markers: deduplicated, uppercase
        - contamination_flag: True if organism in contamination_terms
        - raw_text: stored as-is, NEVER forwarded to MedGemma

    Raises:
        ExtractionError: if both organism AND cfu fail to parse.
    """
    organism = _parse_organism(report_text)
    cfu, cfu_ok = _parse_cfu(report_text)

    if organism is None and not cfu_ok:
        raise ExtractionError(
            "Extraction failed: could not parse organism OR CFU/mL from report. "
            "Check report format."
        )

    # If only organism failed, use a placeholder and warn
    if organism is None:
        warnings.warn(
            "Organism could not be parsed; using 'unknown'.", UserWarning, stacklevel=2
        )
        organism = "unknown"

    resistance_markers = _parse_resistance_markers(report_text)
    specimen_type = _parse_specimen(report_text)
    contamination_flag = _is_contamination(organism)
    date = _parse_date(report_text)

    return CultureReport(
        date=date,
        organism=organism,
        cfu=cfu,
        resistance_markers=resistance_markers,
        specimen_type=specimen_type,
        contamination_flag=contamination_flag,
        raw_text=report_text,  # stored; never forwarded to LLM
    )
