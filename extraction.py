"""
CultureSense Extraction Layer
Parses free-text culture reports into typed CultureReport dataclasses.
"""

import json
import re
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple, Any

from data_models import CultureReport, AntibioticSusceptibility
from rules import RULES, normalize_organism


# ---------------------------------------------------------------------------
# Helper: Docling Processing
# ---------------------------------------------------------------------------
def _process_with_docling(input_text: str) -> str:
    """
    Process input text using Docling.

    If input_text is a valid file path, processes that file.
    Otherwise, writes text to a temporary file and processes it.
    Returns the structured markdown text from the document.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        # Silently fail or log debug if needed, but for user-facing, return original text
        # Only warn once if desired, but here we just return
        return input_text

    input_path = Path(input_text)
    try:
        is_file = input_path.exists() and input_path.is_file()
    except OSError:
        # Input text is too long to be a valid file path
        is_file = False

    try:
        converter = DocumentConverter()

        if is_file:
            # Process directly from file path
            result = converter.convert(input_path)
            return result.document.export_to_markdown()
        else:
            # Input is raw text; Docling processing via temp file may distort layout (e.g. merging lines).
            # Fallback to returning raw text so regexes can use original newlines.
            return input_text

    except Exception as e:
        warnings.warn(
            f"Docling processing failed: {e}. Falling back to raw text.", UserWarning
        )
        return input_text


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class ExtractionError(ValueError):
    """Raised when both organism AND cfu fail to parse from a report."""


# ---------------------------------------------------------------------------
# Compiled regex patterns (Section 5.2) - ENHANCED for flexibility
# ---------------------------------------------------------------------------

# Organism: Multiple patterns to handle various lab report formats
# Fixed: Use greedy match that captures until newline but handles dots in names like "E. coli"
_RE_ORGANISM_PRIMARY = re.compile(r"Organism:\s*([^.].*?)(?:\n|$)", re.IGNORECASE)
_RE_ORGANISM_ALT1 = re.compile(
    r"Organism\s+identified:\s*([^.].*?)(?:\n|$)", re.IGNORECASE
)
_RE_ORGANISM_ALT2 = re.compile(r"Isolated:\s*([^.].*?)(?:\n|$)", re.IGNORECASE)
_RE_ORGANISM_ALT3 = re.compile(r"Identification:\s*([^.].*?)(?:\n|$)", re.IGNORECASE)
_RE_ORGANISM_ALT4 = re.compile(
    r"Culture\s+results?:\s*([^.].*?)(?:\n|$)", re.IGNORECASE
)
_RE_ORGANISM_ALT5 = re.compile(r"ORGANISM:\s*([^.].*?)(?:\n|$)", re.IGNORECASE)

# CFU/mL: Multiple patterns for various formats
_RE_CFU_PRIMARY = re.compile(r"CFU[/\\]?m?L?:\s*([><]?\s*[\d,]+)", re.IGNORECASE)
_RE_CFU_ALT1 = re.compile(
    r"(?:Count|Quantity|Result):\s*([><]?\s*[\d,]+)", re.IGNORECASE
)
# Note: Negative lookbehind for "<", "&lt;", digits, or comma to avoid matching threshold values
# like "<5,000 CFU/mL" or "&lt;5,000 CFU/mL" (HTML-escaped) or partial numbers like ",000"
_RE_CFU_ALT2 = re.compile(r"(?<![<\d,;])(\d[\d,]*)\s*(?:CFU|colonies|cells)", re.IGNORECASE)
_RE_CFU_ALT3 = re.compile(r">\s*?([\d,]+)", re.IGNORECASE)  # >100,000
_RE_CFU_ALT4 = re.compile(r"(\d{1,3},\d{3})", re.IGNORECASE)  # 5,000 or 100,000 pattern

# Fallback CFU patterns
_RE_CFU_SCIENTIFIC = re.compile(r"10\^(\d+)", re.IGNORECASE)  # 10^5 → 100000
_RE_CFU_WORD = re.compile(r"(TNTC|Too\s+Numerous\s+To\s+Count)", re.IGNORECASE)
_RE_CFU_NO_GROWTH = re.compile(
    r"(No\s+growth|No\s+significant\s+growth|0\s+CFU|Negative)", re.IGNORECASE
)
_RE_CFU_RAW_NUMBER = re.compile(r"\b([\d]{5,})\b")  # bare large number (5+ digits)

# Date: Multiple patterns for various formats
_RE_DATE_PRIMARY = re.compile(
    r"(?:Date|Collected|Reported|Specimen\s+Date|Collection\s+Date|Date\s+Collected|Date\s+Reported)[\s:]*[\*_]*[\s:]+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})",
    re.IGNORECASE,
)
_RE_DATE_ALT1 = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")  # ISO format anywhere
_RE_DATE_ALT2 = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")  # MM/DD/YYYY anywhere
_RE_DATE_ALT3 = re.compile(r"\b(\d{2}-\d{2}-\d{4})\b")  # MM-DD-YYYY anywhere

# Resistance markers: exact case-insensitive word boundaries
_RE_RESISTANCE = re.compile(r"\b(ESBL|CRE|MRSA|VRE|CRKP)\b", re.IGNORECASE)

# Susceptibility table patterns
_RE_SUSCEPTIBILITY_ROW = re.compile(
    r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(Sensitive|Intermediate|Resistant|S|I|R)\s*\|\s*([^|]*)\|\s*([^|]*)\|',
    re.IGNORECASE
)

_RE_SUSCEPTIBILITY_ALT = re.compile(
    r'(?:Antibiotic|Antimicrobial|Agent)[\s:]+([^\n]+?)[\s,]+(?:MIC)?[\s:]*([\d<>.=\s]+(?:ug/mL|mcg/mL|mg/L)?)[\s,]+(?:Interpretation)?[\s:]*(S|I|R|Sensitive|Intermediate|Resistant)',
    re.IGNORECASE
)

_RE_ANTIBIOTIC_LINE = re.compile(
    r'^\s*([A-Za-z\s\-]+?)\s+([<>=\d\.]+\s*(?:ug/ml|mcg/ml|mg/l)?)\s+(S|I|R|Sensitive|Intermediate|Resistant)\b',
    re.IGNORECASE | re.MULTILINE
)

# Negation words to check around resistance markers (for context-aware extraction)
_NEGATION_WORDS = ["no ", "not ", "none", "without", "negative for", "undetected", "ruled out"]

# Specimen type - ENHANCED: multiple patterns and keyword detection
_RE_SPECIMEN_PRIMARY = re.compile(
    r"(?:Specimen|Sample|Source|Type)[\s:]+(urine|stool|wound|blood|urinary|fecal|faecal)",
    re.IGNORECASE,
)
_RE_SPECIMEN_ALT1 = re.compile(
    r"(urine|stool|wound|blood)\s*(?:culture|specimen|sample|test)", re.IGNORECASE
)
_RE_SPECIMEN_ALT2 = re.compile(
    r"(?:culture|specimen|sample|test)\s*(?:type)?[\s:]+(urine|stool|wound|blood)",
    re.IGNORECASE,
)
# Match markdown headers and bold text: ## Urine Culture, **Urine Culture**, Urine Culture
_RE_SPECIMEN_HEADER = re.compile(
    r"(?:^#{1,3}\s*|\*{2}|\_{2}|##\s*)\s*(urine|stool|wound|blood|sputum)\s+culture\b",
    re.IGNORECASE | re.MULTILINE,
)
# Quest Diagnostics table format: | Specimen Type | Urine |
_RE_SPECIMEN_TABLE_CELL = re.compile(
    r"\|\s*Specimen\s+(?:Type|Source)\s*\|\s*(urine|stool|wound|blood)\s*\|",
    re.IGNORECASE,
)
_RE_SPECIMEN_URINE_KEYWORD = re.compile(
    r"\b(urine|urinary|bladder|catheter)\b", re.IGNORECASE
)
_RE_SPECIMEN_STOOL_KEYWORD = re.compile(
    r"\b(stool|fecal|faecal|feces|gi)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# CFU normalisation helper (Section 5.4) - ENHANCED
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
        - ">100,000" or "> 100,000"         → 100000 (or parse the number)
        - comma-separated integer           → int (commas stripped)
        - Missing/unparseable               → 0 with warning
    """
    text = report_text.strip()

    # 1. Primary: "CFU/mL: 120,000" or "CFU/mL: >100,000"
    m = _RE_CFU_PRIMARY.search(text)
    if m:
        raw = m.group(1).replace(",", "").replace(">", "").replace("<", "").strip()
        try:
            return int(raw), True
        except ValueError:
            pass

    # 2. Alternative: "Count: 120,000" or "Result: >100,000"
    m = _RE_CFU_ALT1.search(text)
    if m:
        raw = m.group(1).replace(",", "").replace(">", "").replace("<", "").strip()
        try:
            return int(raw), True
        except ValueError:
            pass

    # 3. Alternative: "120,000 CFU" or "120,000 colonies"
    m = _RE_CFU_ALT2.search(text)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            return int(raw), True
        except ValueError:
            pass

    # 4. Alternative: ">100,000" or "> 100,000"
    m = _RE_CFU_ALT3.search(text)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            return int(raw), True
        except ValueError:
            pass

    # 5. Alternative: standalone 100,000 pattern
    m = _RE_CFU_ALT4.search(text)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            return int(raw), True
        except ValueError:
            pass

    # 6. TNTC
    if _RE_CFU_WORD.search(text):
        return 999999, True

    # 7. No growth / negative
    if _RE_CFU_NO_GROWTH.search(text):
        return 0, True

    # 8. Scientific notation "10^5"
    m = _RE_CFU_SCIENTIFIC.search(text)
    if m:
        try:
            return 10 ** int(m.group(1)), True
        except (ValueError, OverflowError):
            pass

    # 9. Bare large integer (≥5 digits) — last resort fallback
    m = _RE_CFU_RAW_NUMBER.search(text)
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
    # Look for "Collected:" pattern first (most reliable indicator of collection date)
    collected_pattern = re.compile(
        r"Collected:\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})",
        re.IGNORECASE
    )
    m = collected_pattern.search(report_text)
    if m:
        raw = m.group(1)
        return _normalize_date(raw)

    # Primary: prefixed dates (Date:, Date Collected:, etc.)
    m = _RE_DATE_PRIMARY.search(report_text)
    if m:
        raw = m.group(1)
        return _normalize_date(raw)

    # Alt1: ISO format anywhere (but skip if it looks like a birth date)
    all_dates = _RE_DATE_ALT1.findall(report_text)
    if all_dates:
        # If there's a DATE OF BIRTH field, try to exclude dates near it
        if "DATE OF BIRTH" in report_text.upper():
            # Find all ISO dates and their positions
            for date in all_dates:
                pos = report_text.find(date)
                birth_pos = report_text.upper().find("DATE OF BIRTH")
                # If date is far from DATE OF BIRTH, it's likely collection date
                if abs(pos - birth_pos) > 50:
                    return date
            # If all dates are near birth date, return unknown
            return "unknown"
        return all_dates[0]

    # Alt2: MM/DD/YYYY anywhere
    m = _RE_DATE_ALT2.search(report_text)
    if m:
        return _normalize_date(m.group(1))

    # Alt3: MM-DD-YYYY anywhere
    m = _RE_DATE_ALT3.search(report_text)
    if m:
        raw = m.group(1).replace("-", "/")
        return _normalize_date(raw)

    return "unknown"


def _normalize_date(raw: str) -> str:
    """Convert various date formats to ISO 8601 (YYYY-MM-DD)."""
    raw = raw.strip()

    # Already ISO format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    # MM/DD/YYYY or MM-DD-YYYY
    if "/" in raw or "-" in raw:
        sep = "/" if "/" in raw else "-"
        parts = raw.split(sep)
        if len(parts) == 3:
            # Determine if first part is month or day based on values
            first, second, year = parts[0], parts[1], parts[2]
            # If first > 12, it's likely DD/MM/YYYY
            if int(first) > 12:
                # DD/MM/YYYY → YYYY-MM-DD
                return f"{year}-{second.zfill(2)}-{first.zfill(2)}"
            else:
                # MM/DD/YYYY → YYYY-MM-DD
                return f"{year}-{first.zfill(2)}-{second.zfill(2)}"

    return "unknown"


def _parse_organism(report_text: str) -> Optional[str]:
    """
    Extract organism name from report text with multiple pattern attempts.
    """
    text = report_text.strip()

    # Try multiple organism patterns in order
    patterns = [
        _RE_ORGANISM_PRIMARY,
        _RE_ORGANISM_ALT5,  # ORGANISM: (all caps)
        _RE_ORGANISM_ALT1,  # Organism identified:
        _RE_ORGANISM_ALT2,  # Isolated:
        _RE_ORGANISM_ALT3,  # Identification:
        _RE_ORGANISM_ALT4,  # Culture result:
    ]

    for pattern in patterns:
        m = pattern.search(text)
        if m:
            raw_organism = m.group(1).strip()
            # Clean up common artifacts but preserve dots in organism names like "E. coli"
            raw_organism = re.sub(r"\s+", " ", raw_organism)  # normalize whitespace
            # Don't split on dots - they're part of organism names like "E. coli"
            # Only truncate if there's clear sentence-ending punctuation
            if re.search(r"[;!?]|\.\s+[A-Z]", raw_organism):
                # Find the first sentence-ending punctuation
                match = re.search(r"([;!?]|\.\s+[A-Z])", raw_organism)
                if match:
                    raw_organism = raw_organism[: match.start()]
            return normalize_organism(raw_organism)

    # Fallback: search for known organism aliases in full text
    lower_text = text.lower()
    from rules import ORGANISM_ALIASES

    for alias in sorted(ORGANISM_ALIASES.keys(), key=len, reverse=True):
        if alias in lower_text:
            return normalize_organism(alias)

    return None


def _parse_resistance_markers(report_text: str) -> list[str]:
    """Extract all high-risk resistance markers (deduplicated, uppercase)."""
    found = []
    for match in _RE_RESISTANCE.finditer(report_text):
        marker = match.group(1)
        # Check 60-char window around match for negation
        start = max(0, match.start() - 60)
        end = min(len(report_text), match.end() + 60)
        context = report_text[start:end].lower()
        if any(neg in context for neg in _NEGATION_WORDS):
            continue  # Skip this match - it's in a negation context
        found.append(marker)
    # deduplicate, preserve order
    return list(dict.fromkeys(m.upper() for m in found))


def _parse_susceptibility_profile(report_text: str) -> list[AntibioticSusceptibility]:
    """
    Extract antimicrobial susceptibility profile from report text.

    Parses susceptibility tables in various formats:
    - Markdown table format: | Antibiotic | MIC | S/I/R | Breakpoints |
    - Simple format: Antibiotic: MIC (S/I/R)

    Returns a list of AntibioticSusceptibility dataclass instances.
    """
    profile: list[AntibioticSusceptibility] = []
    seen_antibiotics: set[str] = set()

    # Pattern 1: Markdown table rows | Antibiotic | MIC | Interpretation | ...
    for match in _RE_SUSCEPTIBILITY_ROW.finditer(report_text):
        antibiotic = match.group(1).strip()
        mic = match.group(2).strip()
        interp_raw = match.group(3).strip().upper()
        breakpoints = match.group(4).strip() if len(match.groups()) >= 4 else ""
        notes = match.group(5).strip() if len(match.groups()) >= 5 else ""

        # Normalize interpretation to S/I/R
        if interp_raw in ("S", "SENSITIVE"):
            interpretation = "S"
        elif interp_raw in ("I", "INTERMEDIATE"):
            interpretation = "I"
        elif interp_raw in ("R", "RESISTANT"):
            interpretation = "R"
        else:
            interpretation = interp_raw

        # Skip if not a valid antibiotic name (too short or looks like a header)
        if len(antibiotic) < 3 or antibiotic.lower() in ("antibiotic", "agent", "drug", "name"):
            continue

        # Deduplicate
        antibiotic_lower = antibiotic.lower()
        if antibiotic_lower in seen_antibiotics:
            continue
        seen_antibiotics.add(antibiotic_lower)

        profile.append(AntibioticSusceptibility(
            antibiotic=antibiotic,
            mic=mic,
            interpretation=interpretation,
            breakpoints=breakpoints,
            notes=notes
        ))

    # Pattern 2: Alternative format (Antibiotic, MIC, Interpretation inline)
    for match in _RE_SUSCEPTIBILITY_ALT.finditer(report_text):
        antibiotic = match.group(1).strip()
        mic = match.group(2).strip() if len(match.groups()) >= 2 else ""
        interp_raw = match.group(3).strip().upper() if len(match.groups()) >= 3 else ""

        if interp_raw in ("S", "SENSITIVE"):
            interpretation = "S"
        elif interp_raw in ("I", "INTERMEDIATE"):
            interpretation = "I"
        elif interp_raw in ("R", "RESISTANT"):
            interpretation = "R"
        else:
            continue  # Skip if no valid interpretation

        if len(antibiotic) < 3 or antibiotic.lower() in ("antibiotic", "agent", "drug", "name"):
            continue

        antibiotic_lower = antibiotic.lower()
        if antibiotic_lower in seen_antibiotics:
            continue
        seen_antibiotics.add(antibiotic_lower)

        profile.append(AntibioticSusceptibility(
            antibiotic=antibiotic,
            mic=mic,
            interpretation=interpretation,
            breakpoints="",
            notes=""
        ))

    # Pattern 3: Simple line format
    for match in _RE_ANTIBIOTIC_LINE.finditer(report_text):
        antibiotic = match.group(1).strip()
        mic = match.group(2).strip()
        interp_raw = match.group(3).strip().upper()

        if interp_raw in ("S", "SENSITIVE"):
            interpretation = "S"
        elif interp_raw in ("I", "INTERMEDIATE"):
            interpretation = "I"
        elif interp_raw in ("R", "RESISTANT"):
            interpretation = "R"
        else:
            interpretation = interp_raw

        if len(antibiotic) < 3 or antibiotic.lower() in ("antibiotic", "agent", "drug", "name"):
            continue

        antibiotic_lower = antibiotic.lower()
        if antibiotic_lower in seen_antibiotics:
            continue
        seen_antibiotics.add(antibiotic_lower)

        profile.append(AntibioticSusceptibility(
            antibiotic=antibiotic,
            mic=mic,
            interpretation=interpretation,
            breakpoints="",
            notes=""
        ))

    return profile


def _format_susceptibility_summary(profile: list[AntibioticSusceptibility]) -> str:
    """Format susceptibility profile as a concise summary string."""
    if not profile:
        return ""

    s_count = sum(1 for a in profile if a.interpretation == "S")
    i_count = sum(1 for a in profile if a.interpretation == "I")
    r_count = sum(1 for a in profile if a.interpretation == "R")

    total = len(profile)
    return f"{total} antibiotics: {s_count}S/{i_count}I/{r_count}R"


def _parse_specimen(report_text: str) -> str:
    """
    Extract specimen type with multiple pattern attempts and keyword detection.
    Returns 'urine', 'stool', 'wound', 'blood', or 'unknown'.
    """
    text = report_text.strip()

    # Try markdown headers and bold text: ## Urine Culture, **Urine Culture**
    m = _RE_SPECIMEN_HEADER.search(text)
    if m:
        return _normalize_specimen(m.group(1).lower())

    # Try table cell format: | Specimen Type | Urine | (Quest Diagnostics format)
    m = _RE_SPECIMEN_TABLE_CELL.search(text)
    if m:
        return _normalize_specimen(m.group(1).lower())

    # Try primary pattern: Specimen/Sample/Source/Type: urine/stool
    m = _RE_SPECIMEN_PRIMARY.search(text)
    if m:
        specimen = m.group(1).lower()
        return _normalize_specimen(specimen)

    # Try alternative: urine/stool culture
    m = _RE_SPECIMEN_ALT1.search(text)
    if m:
        return _normalize_specimen(m.group(1).lower())

    # Try alternative: culture: urine/stool
    m = _RE_SPECIMEN_ALT2.search(text)
    if m:
        return _normalize_specimen(m.group(1).lower())

    # Keyword detection: look for urine/urinary keywords anywhere
    if _RE_SPECIMEN_URINE_KEYWORD.search(text):
        return "urine"

    # Keyword detection: look for stool/fecal keywords anywhere
    if _RE_SPECIMEN_STOOL_KEYWORD.search(text):
        return "stool"

    return "unknown"


def _normalize_specimen(specimen: str) -> str:
    """Normalize specimen type to standard values."""
    specimen = specimen.lower().strip()

    # Map variations to standard types
    if specimen in ("urine", "urinary"):
        return "urine"
    elif specimen in ("stool", "fecal", "faecal", "feces"):
        return "stool"
    elif specimen == "wound":
        return "wound"
    elif specimen == "blood":
        return "blood"

    return specimen


def _is_contamination(organism: str) -> bool:
    """Return True if the organism name matches any contamination term."""
    lower = organism.lower()
    return any(term in lower for term in RULES["contamination_terms"])


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------


def debug_extraction(report_text: str, label: str = "Report") -> dict:
    """
    Debug helper to show what was extracted from a report.

    Returns a dictionary with all extraction results for debugging.
    """
    try:
        is_file = Path(report_text).exists()
    except OSError:
        is_file = False
    processed_text = (
        _process_with_docling(report_text)
        if is_file
        else report_text
    )

    organism = _parse_organism(processed_text)
    cfu, cfu_ok = _parse_cfu(processed_text)
    specimen = _parse_specimen(processed_text)
    date = _parse_date(processed_text)
    resistance = _parse_resistance_markers(processed_text)
    susceptibility = _parse_susceptibility_profile(processed_text)

    return {
        "label": label,
        "organism": organism,
        "cfu": cfu,
        "cfu_ok": cfu_ok,
        "specimen": specimen,
        "date": date,
        "resistance": resistance,
        "susceptibility": susceptibility,
        "is_contamination": _is_contamination(organism) if organism else False,
        "processed_text_preview": processed_text[:500] + "..."
        if len(processed_text) > 500
        else processed_text,
    }


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------


def extract_structured_data(report_text: str) -> CultureReport:
    """
    Parse a free-text culture report into a typed CultureReport.

    Now supports direct file paths via Docling processing.

    Rules:
        - Organism field: stripped, normalised via ORGANISM_ALIASES
        - CFU: commas removed, converted to int; TNTC=999999
        - resistance_markers: deduplicated, uppercase
        - contamination_flag: True if organism in contamination_terms
        - raw_text: stored as-is (or docling processed), NEVER forwarded to MedGemma

    Raises:
        ExtractionError: if both organism AND cfu fail to parse.
    """
    # Pre-process with Docling (handles file paths or raw text)
    processed_text = _process_with_docling(report_text)

    # Attempt extraction on processed text
    organism = _parse_organism(processed_text)
    cfu, cfu_ok = _parse_cfu(processed_text)

    # Fallback: if extraction failed and text was modified by Docling, try original
    if (organism is None and not cfu_ok) and processed_text != report_text:
        organism = _parse_organism(report_text)
        cfu, cfu_ok = _parse_cfu(report_text)
        if organism is not None or cfu_ok:
            processed_text = report_text  # Revert to original for other fields

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

    resistance_markers = _parse_resistance_markers(processed_text)
    specimen_type = _parse_specimen(processed_text)
    contamination_flag = _is_contamination(organism)
    date = _parse_date(processed_text)
    susceptibility_profile = _parse_susceptibility_profile(processed_text)

    return CultureReport(
        date=date,
        organism=organism,
        cfu=cfu,
        resistance_markers=resistance_markers,
        susceptibility_profile=susceptibility_profile,
        specimen_type=specimen_type,
        contamination_flag=contamination_flag,
        raw_text=processed_text,  # Store the text actually used for extraction
    )


# =============================================================================
# MedGemma Fallback Extraction
# =============================================================================

def _build_medgemma_extraction_prompt(report_text: str) -> str:
    """
    Build a structured prompt for MedGemma to extract culture report fields.

    The prompt asks MedGemma to extract specific fields in JSON format.
    This is used as a fallback when regex extraction fails.
    """
    # Truncate text if too long to avoid token limits
    truncated_text = report_text[:2000] if len(report_text) > 2000 else report_text

    prompt = f"""You are a medical data extraction assistant. Extract structured information from the following microbiology culture report.

Return ONLY a valid JSON object with these exact fields:
- "organism": The name of the identified organism (e.g., "E. coli", "Klebsiella pneumoniae", "Mixed flora"). Use "unknown" if not found.
- "cfu": The colony forming units per mL as an integer (e.g., 100000). Use 0 if not found or for "No growth".
- "date": The collection date in YYYY-MM-DD format. Use "unknown" if not found.
- "specimen_type": Either "urine", "stool", or "unknown".
- "resistance_markers": List of resistance markers found (e.g., ["ESBL", "MRSA"]). Use empty list [] if none.

Culture Report Text:
---
{truncated_text}
---

JSON Output:"""
    return prompt


def _parse_medgemma_extraction_response(response: str) -> dict:
    """
    Parse MedGemma's JSON response into a dictionary.

    Handles common JSON formatting issues from LLM outputs.
    """
    # Try to extract JSON from the response
    # Sometimes LLMs wrap JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        response = json_match.group(1)

    # Try to find raw JSON object
    json_match = re.search(r'\{[\s\S]*"organism"[\s\S]*\}', response)
    if json_match:
        response = json_match.group(0)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Fallback: try to extract key-value pairs manually
        data = {}
        for key in ["organism", "cfu", "date", "specimen_type", "resistance_markers"]:
            pattern = rf'"{key}"\s*:\s*([^,\}}]+)'
            match = re.search(pattern, response)
            if match:
                value = match.group(1).strip().strip('"')
                if key == "cfu":
                    try:
                        data[key] = int(value)
                    except ValueError:
                        data[key] = 0
                elif key == "resistance_markers":
                    # Parse list format
                    if value.startswith("["):
                        try:
                            data[key] = json.loads(value)
                        except:
                            data[key] = []
                    else:
                        data[key] = [v.strip().strip('"') for v in value.split(",") if v.strip()]
                else:
                    data[key] = value

    # Validate and set defaults
    if "organism" not in data or not data["organism"]:
        data["organism"] = "unknown"
    if "cfu" not in data:
        data["cfu"] = 0
    if "date" not in data or not data["date"]:
        data["date"] = "unknown"
    if "specimen_type" not in data or not data["specimen_type"]:
        data["specimen_type"] = "unknown"
    if "resistance_markers" not in data:
        data["resistance_markers"] = []

    return data


def extract_structured_data_with_fallback(
    report_text: str,
    medgemma_model=None,
    medgemma_tokenizer=None,
    use_medgemma_fallback: bool = True
) -> CultureReport:
    """
    Extract structured data from a culture report with MedGemma fallback.

    This function first attempts regex-based extraction. If that fails (ExtractionError),
    it optionally falls back to MedGemma for LLM-based extraction.

    Args:
        report_text: The raw culture report text
        medgemma_model: The MedGemma model (required for fallback)
        medgemma_tokenizer: The MedGemma tokenizer (required for fallback)
        use_medgemma_fallback: Whether to use MedGemma when regex fails

    Returns:
        A CultureReport dataclass with extracted fields

    Raises:
        ExtractionError: If both regex and MedGemma extraction fail
    """
    # First, try regex-based extraction
    try:
        return extract_structured_data(report_text)
    except ExtractionError as e:
        if not use_medgemma_fallback or medgemma_model is None or medgemma_tokenizer is None:
            # No fallback available, re-raise the original error
            raise e

        # Fall back to MedGemma extraction
        import warnings
        warnings.warn(
            "Regex extraction failed, attempting MedGemma fallback extraction.",
            UserWarning,
            stacklevel=2
        )

        try:
            return _extract_with_medgemma(
                report_text, medgemma_model, medgemma_tokenizer
            )
        except Exception as medgemma_error:
            # Both methods failed
            raise ExtractionError(
                f"Extraction failed: regex extraction failed ({e}) and "
                f"MedGemma fallback also failed ({medgemma_error})."
            )


def _extract_with_medgemma(
    report_text: str,
    model,
    tokenizer
) -> CultureReport:
    """
    Use MedGemma to extract structured data from a culture report.

    This is an internal fallback function used when regex extraction fails.
    """
    import torch

    # Build the extraction prompt
    prompt = _build_medgemma_extraction_prompt(report_text)

    # Generate response from MedGemma
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the response
    if prompt in response:
        response = response[len(prompt):].strip()

    # Parse the JSON response
    extracted = _parse_medgemma_extraction_response(response)

    # Build and return the CultureReport
    organism = normalize_organism(extracted.get("organism", "unknown"))
    cfu = int(extracted.get("cfu", 0))
    date = extracted.get("date", "unknown")
    specimen_type = extracted.get("specimen_type", "unknown")
    resistance_markers = extracted.get("resistance_markers", [])

    # Normalize resistance markers
    valid_markers = {"ESBL", "CRE", "MRSA", "VRE", "CRKP"}
    resistance_markers = [
        m.upper() for m in resistance_markers
        if m.upper() in valid_markers
    ]

    contamination_flag = any(
        term in organism.lower() for term in RULES["contamination_terms"]
    )

    return CultureReport(
        date=date,
        organism=organism,
        cfu=cfu,
        resistance_markers=resistance_markers,
        susceptibility_profile=[],  # MedGemma fallback doesn't extract susceptibility
        specimen_type=specimen_type,
        contamination_flag=contamination_flag,
        raw_text="",  # Never store raw text when using MedGemma fallback
    )
