"""
CultureSense PII/PHI Removal Layer

Removes all personally identifiable information and protected health information
from markdown text before processing. This is a critical safety layer that
ensures patient data is never forwarded to MedGemma or stored in logs.

Usage:
    from pii_removal import scrub_pii, detect_pii

    clean_text = scrub_pii(raw_markdown)
    pii_types_found = detect_pii(raw_markdown)
"""

import re
from typing import List


# -----------------------------------------------------------------------------
# PII Pattern Definitions
# -----------------------------------------------------------------------------

# Patient name patterns - match common name labels followed by name-like text
# KEY PRINCIPLE: Patterns must stop at end of line to avoid over-matching
_NAME_PATTERNS = [
    # Patient Name: John Smith (captures until end of line)
    (re.compile(r"Patient\s*Name\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Patient Name: [REDACTED NAME]"),
    # Patient: Jane Doe
    (re.compile(r"^Patient\s*[:\-]\s*[^\n]*", re.IGNORECASE | re.MULTILINE), "Patient: [REDACTED NAME]"),
    # Pt Name: John Smith
    (re.compile(r"Pt\.?\s*Name\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Pt Name: [REDACTED NAME]"),
    # Pt: Jane Doe
    (re.compile(r"^Pt\.?\s*[:\-]\s*[^\n]*", re.IGNORECASE | re.MULTILINE), "Pt: [REDACTED NAME]"),
    # Name: John Smith (when standalone, avoid matching "Organism name:" etc)
    (re.compile(r"^Name\s*[:\-]\s*[A-Z][^\n]*", re.MULTILINE), "Name: [REDACTED NAME]"),
]

# Date of Birth patterns
_DOB_PATTERNS = [
    # DOB: various formats
    (re.compile(r"DOB\s*[:\-]\s*[^\n]*", re.IGNORECASE), "DOB: [REDACTED DOB]"),
    # Date of Birth: various formats
    (re.compile(r"Date\s+of\s+Birth\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Date of Birth: [REDACTED DOB]"),
    # Birth Date: format
    (re.compile(r"Birth\s*Date\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Birth Date: [REDACTED DOB]"),
    # Born: format
    (re.compile(r"^Born\s*[:\-]\s*[^\n]*", re.IGNORECASE | re.MULTILINE), "Born: [REDACTED DOB]"),
]

# Medical Record Number patterns
_MRN_PATTERNS = [
    # MRN: alphanumeric value
    (re.compile(r"MRN\s*[:\-#]?\s*[^\n]*", re.IGNORECASE), "MRN: [REDACTED MRN]"),
    # Medical Record Number: value
    (re.compile(r"Medical\s+Record\s*(?:Number|No|#)?\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Medical Record Number: [REDACTED MRN]"),
    # MR #: value
    (re.compile(r"MR\s*#\s*[:\-]?\s*[^\n]*", re.IGNORECASE), "MR #: [REDACTED MRN]"),
    # Account #: value
    (re.compile(r"Account\s*(?:Number|No|#)?\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Account #: [REDACTED MRN]"),
    # Patient ID: value
    (re.compile(r"Patient\s*ID\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Patient ID: [REDACTED MRN]"),
    # Encounter #: value
    (re.compile(r"Encounter\s*(?:Number|No|#)?\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Encounter #: [REDACTED MRN]"),
    # Visit #: value
    (re.compile(r"Visit\s*(?:Number|No|#)?\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Visit #: [REDACTED MRN]"),
]

# Social Security Number patterns
_SSN_PATTERNS = [
    # SSN: XXX-XX-XXXX or XXXXXXXXX
    (re.compile(r"SSN\s*[:\-]?\s*[^\n]*", re.IGNORECASE), "SSN: [REDACTED SSN]"),
    # Social Security Number: various formats
    (re.compile(r"Social\s+Security\s*(?:Number|No)?\s*[:\-]?\s*[^\n]*", re.IGNORECASE), "Social Security Number: [REDACTED SSN]"),
]

# Phone number patterns
_PHONE_PATTERNS = [
    # Phone: (XXX) XXX-XXXX
    (re.compile(r"(?:Phone|Tel|Telephone|Mobile|Cell|Fax)\s*[:\-]?\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", re.IGNORECASE), "[REDACTED PHONE]"),
    # Standalone phone numbers in common formats (with word boundaries)
    (re.compile(r"\b\d{3}[.-]\d{3}[.-]\d{4}\b"), "[REDACTED PHONE]"),
    (re.compile(r"\(\d{3}\)\s*\d{3}[.-]?\d{4}\b"), "[REDACTED PHONE]"),
]

# Email address patterns
_EMAIL_PATTERNS = [
    # Email: prefix
    (re.compile(r"(?:Email|E-mail)\s*[:\-]?\s*[^\n]*@[^\n]*", re.IGNORECASE), "Email: [REDACTED EMAIL]"),
    # Standalone emails (not preceded by label)
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[REDACTED EMAIL]"),
]

# Address patterns
_ADDRESS_PATTERNS = [
    # Address: street address (single line, captures until end)
    (re.compile(r"(?:Address|Street|Addr)\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Address: [REDACTED ADDRESS]"),
]

# Provider name patterns (optional - may be disabled)
_PROVIDER_PATTERNS = [
    # Provider: Dr. Name | Physician: Name
    (re.compile(r"(?:Provider|Physician|Doctor|Ordering\s+Physician|Attending|Referring)\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Provider: [REDACTED PROVIDER]"),
    # Dr.: Name
    (re.compile(r"\bDr\.?\s*[:\-]\s*[^\n]*", re.IGNORECASE), "Dr.: [REDACTED PROVIDER]"),
    # Ordered by: Dr. Name
    (re.compile(r"Ordered\s+(?:by|from)\s*[:\-]?\s*[^\n]*", re.IGNORECASE), "Ordered by: [REDACTED PROVIDER]"),
]


# -----------------------------------------------------------------------------
# Detection-only patterns (for reporting what was found)
# -----------------------------------------------------------------------------

_DETECTION_PATTERNS = {
    "name": [
        re.compile(r"Patient\s*Name\s*[:\-]", re.IGNORECASE),
        re.compile(r"^Patient\s*[:\-]", re.IGNORECASE | re.MULTILINE),
        re.compile(r"Pt\.?\s*Name\s*[:\-]", re.IGNORECASE),
        re.compile(r"^Pt\.?\s*[:\-]", re.IGNORECASE | re.MULTILINE),
    ],
    "dob": [
        re.compile(r"DOB\s*[:\-]", re.IGNORECASE),
        re.compile(r"Date\s+of\s+Birth", re.IGNORECASE),
        re.compile(r"Birth\s*Date\s*[:\-]", re.IGNORECASE),
        re.compile(r"^Born\s*[:\-]", re.IGNORECASE | re.MULTILINE),
    ],
    "mrn": [
        re.compile(r"MRN\s*[:\-#]?", re.IGNORECASE),
        re.compile(r"Medical\s+Record", re.IGNORECASE),
        re.compile(r"\bMR\s*#", re.IGNORECASE),
        re.compile(r"Account\s*(?:Number|No|#)?\s*[:\-]", re.IGNORECASE),
        re.compile(r"Patient\s*ID\s*[:\-]", re.IGNORECASE),
        re.compile(r"Encounter\s*(?:Number|No|#)?\s*[:\-]", re.IGNORECASE),
        re.compile(r"Visit\s*(?:Number|No|#)?\s*[:\-]", re.IGNORECASE),
    ],
    "ssn": [
        re.compile(r"SSN\s*[:\-]?", re.IGNORECASE),
        re.compile(r"Social\s+Security", re.IGNORECASE),
    ],
    "phone": [
        re.compile(r"(?:Phone|Tel|Telephone|Mobile|Cell|Fax)\s*[:\-]", re.IGNORECASE),
        re.compile(r"\b\d{3}[.-]\d{3}[.-]\d{4}\b"),
        re.compile(r"\(\d{3}\)\s*\d{3}[.-]?\d{4}\b"),
    ],
    "email": [
        re.compile(r"(?:Email|E-mail)\s*[:\-]", re.IGNORECASE),
        re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    ],
    "address": [
        re.compile(r"(?:Address|Street|Addr)\s*[:\-]", re.IGNORECASE),
    ],
    "provider": [
        re.compile(r"(?:Provider|Physician|Doctor|Dr)\s*[:\-]", re.IGNORECASE),
        re.compile(r"Ordered\s+(?:by|from)", re.IGNORECASE),
    ],
}


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def scrub_pii(markdown_text: str, remove_provider_names: bool = False) -> str:
    """
    Remove all PII/PHI from markdown text.

    Applies regex-based scrubbing for:
    - Patient names
    - Dates of birth
    - Medical record numbers
    - Social security numbers
    - Phone numbers
    - Email addresses
    - Street addresses
    - Provider names (optional)

    Args:
        markdown_text: Raw text from Docling PDF extraction
        remove_provider_names: If True, also scrub provider/doctor names

    Returns:
        Text with all PII replaced with [REDACTED ...] markers
    """
    if not markdown_text:
        return ""

    text = markdown_text

    # Apply each pattern set
    for pattern, replacement in _NAME_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in _DOB_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in _MRN_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in _SSN_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in _PHONE_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in _EMAIL_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in _ADDRESS_PATTERNS:
        text = pattern.sub(replacement, text)

    if remove_provider_names:
        for pattern, replacement in _PROVIDER_PATTERNS:
            text = pattern.sub(replacement, text)

    return text


def detect_pii(markdown_text: str) -> List[str]:
    """
    Detect what types of PII are present in the text.

    Returns a list of PII type identifiers found:
    - "name" - Patient names detected
    - "dob" - Date of birth detected
    - "mrn" - Medical record number detected
    - "ssn" - Social security number detected
    - "phone" - Phone number detected
    - "email" - Email address detected
    - "address" - Address detected
    - "provider" - Provider name detected

    This is useful for logging/auditing without logging the actual PII.

    Args:
        markdown_text: Raw text to analyze

    Returns:
        List of PII type strings that were detected
    """
    if not markdown_text:
        return []

    detected = []

    for pii_type, patterns in _DETECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(markdown_text):
                detected.append(pii_type)
                break  # Only count each type once

    return detected


def scrub_pii_debug(markdown_text: str, remove_provider_names: bool = False) -> tuple[str, dict]:
    """
    Scrub PII and return detailed information about what was found.

    Args:
        markdown_text: Raw text from Docling PDF extraction
        remove_provider_names: If True, also scrub provider/doctor names

    Returns:
        Tuple of (scrubbed_text, debug_info_dict)
        debug_info_dict contains:
        - 'types_found': list of PII types detected
        - 'redaction_count': estimated number of redactions made
    """
    if not markdown_text:
        return "", {"types_found": [], "redaction_count": 0}

    types_found = detect_pii(markdown_text)

    # Count approximate redactions before scrubbing
    redaction_count = 0
    all_patterns = (
        _NAME_PATTERNS + _DOB_PATTERNS + _MRN_PATTERNS +
        _SSN_PATTERNS + _PHONE_PATTERNS + _EMAIL_PATTERNS +
        _ADDRESS_PATTERNS
    )
    if remove_provider_names:
        all_patterns += _PROVIDER_PATTERNS

    for pattern, _ in all_patterns:
        matches = pattern.findall(markdown_text)
        redaction_count += len(matches)

    scrubbed = scrub_pii(markdown_text, remove_provider_names)

    debug_info = {
        "types_found": types_found,
        "redaction_count": redaction_count,
    }

    return scrubbed, debug_info


# -----------------------------------------------------------------------------
# Module self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick self-test
    test_text = """Patient Name: John Smith
DOB: 01/15/1980
MRN: 12345678
SSN: 123-45-6789
Phone: (555) 123-4567
Email: john.smith@email.com
Address: 123 Main St, Springfield, IL
Provider: Dr. Sarah Chen

Organism: E. coli
CFU/mL: 100,000"""

    print("Original text:")
    print(test_text)
    print("\n" + "="*50 + "\n")

    detected = detect_pii(test_text)
    print(f"PII types detected: {detected}")

    scrubbed = scrub_pii(test_text, remove_provider_names=True)
    print("\nScrubbed text:")
    print(scrubbed)
