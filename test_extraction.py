"""
CultureSense Extraction Layer — Unit Tests (Cell C-3)
Kaggle-native inline tests (no pytest dependency).
"""

import warnings

from extraction import ExtractionError, extract_structured_data

_PASS = 0
_FAIL = 0


def _assert(condition: bool, msg: str) -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {msg}")
    else:
        _FAIL += 1
        print(f"  FAIL  {msg}")


# ---------------------------------------------------------------------------
# Test Report 1 — Normal improving report
# ---------------------------------------------------------------------------
REPORT_NORMAL = """
Specimen: Urine
Date Collected: 2026-01-01
Organism: E. coli
CFU/mL: 120,000
Sensitivity: Ampicillin - Resistant, Nitrofurantoin - Sensitive
"""

print("=== Test: Normal Report ===")
r = extract_structured_data(REPORT_NORMAL)
_assert(r.date == "2026-01-01", f"date == '2026-01-01'  (got '{r.date}')")
_assert(
    r.organism == "Escherichia coli",
    f"organism normalised to 'Escherichia coli'  (got '{r.organism}')",
)
_assert(r.cfu == 120000, f"cfu == 120000  (got {r.cfu})")
_assert(
    r.resistance_markers == [], f"no resistance markers  (got {r.resistance_markers})"
)
_assert(
    r.specimen_type == "urine", f"specimen_type == 'urine'  (got '{r.specimen_type}')"
)
_assert(
    r.contamination_flag is False,
    f"contamination_flag is False  (got {r.contamination_flag})",
)

# ---------------------------------------------------------------------------
# Test Report 2 — Contamination report (mixed flora, low CFU)
# ---------------------------------------------------------------------------
REPORT_CONTAMINATION = """
Specimen: Urine
Date Collected: 2026-02-05
Organism: mixed flora
CFU/mL: 5,000
No resistance markers detected.
"""

print("\n=== Test: Contamination Report ===")
r2 = extract_structured_data(REPORT_CONTAMINATION)
_assert(
    r2.contamination_flag is True,
    f"contamination_flag is True  (got {r2.contamination_flag})",
)
_assert(
    r2.organism == "mixed flora", f"organism == 'mixed flora'  (got '{r2.organism}')"
)
_assert(r2.cfu == 5000, f"cfu == 5000  (got {r2.cfu})")
_assert(
    r2.resistance_markers == [], f"no resistance markers  (got {r2.resistance_markers})"
)

# ---------------------------------------------------------------------------
# Test Report 3 — Resistance-containing report (ESBL marker)
# ---------------------------------------------------------------------------
REPORT_RESISTANCE = """
Specimen: Urine
Date Collected: 2026-01-20
Organism: Klebsiella pneumoniae
CFU/mL: 75,000
Resistance: ESBL detected.
"""

print("\n=== Test: Resistance Report ===")
r3 = extract_structured_data(REPORT_RESISTANCE)
_assert(
    r3.organism == "Klebsiella pneumoniae",
    f"organism == 'Klebsiella pneumoniae'  (got '{r3.organism}')",
)
_assert(
    "ESBL" in r3.resistance_markers,
    f"ESBL in resistance_markers  (got {r3.resistance_markers})",
)
_assert(
    r3.contamination_flag is False,
    f"contamination_flag is False  (got {r3.contamination_flag})",
)
_assert(r3.cfu == 75000, f"cfu == 75000  (got {r3.cfu})")

# ---------------------------------------------------------------------------
# Test — TNTC CFU normalisation
# ---------------------------------------------------------------------------
REPORT_TNTC = """
Specimen: Urine
Date Collected: 2026-03-01
Organism: E. coli
CFU/mL: TNTC
"""

print("\n=== Test: TNTC Normalisation ===")
r4 = extract_structured_data(REPORT_TNTC)
_assert(r4.cfu == 999999, f"TNTC → 999999  (got {r4.cfu})")

# ---------------------------------------------------------------------------
# Test — No growth / cleared
# ---------------------------------------------------------------------------
REPORT_NO_GROWTH = """
Specimen: Urine
Date Collected: 2026-03-15
Organism: E. coli
No growth observed.
"""

print("\n=== Test: No Growth ===")
r5 = extract_structured_data(REPORT_NO_GROWTH)
_assert(r5.cfu == 0, f"No growth → cfu == 0  (got {r5.cfu})")

# ---------------------------------------------------------------------------
# Test — ExtractionError on completely unparseable input
# ---------------------------------------------------------------------------
print("\n=== Test: ExtractionError on bad input ===")
try:
    extract_structured_data("this report contains absolutely nothing useful at all")
    _assert(False, "ExtractionError should have been raised")
except ExtractionError as e:
    _assert(True, f"ExtractionError raised correctly: {e}")
except Exception as e:
    _assert(False, f"Wrong exception type raised: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Test — Adversarial: SQL injection in CFU field
# ---------------------------------------------------------------------------
REPORT_ADV = """
Specimen: Urine
Date Collected: 2026-04-01
Organism: E. coli
CFU/mL: 100000; DROP TABLE reports
"""

print("\n=== Test: Adversarial SQL Injection in CFU ===")
# Should parse 100000 from the start, or fallback gracefully
try:
    r6 = extract_structured_data(REPORT_ADV)
    # The regex only captures digits+commas, so "100000" is parsed, the rest is ignored
    _assert(r6.cfu == 100000, f"cfu == 100000 (injection ignored)  (got {r6.cfu})")
except ExtractionError:
    _assert(False, "Should not raise ExtractionError on adversarial CFU")

# ---------------------------------------------------------------------------
# Test — Alternate date format MM/DD/YYYY
# ---------------------------------------------------------------------------
REPORT_DATE_ALT = """
Specimen: Stool
Date Collected: 01/15/2026
Organism: Enterococcus faecalis
CFU/mL: 60,000
"""

print("\n=== Test: Alternate Date Format (MM/DD/YYYY) ===")
r7 = extract_structured_data(REPORT_DATE_ALT)
_assert(r7.date == "2026-01-15", f"date normalised to ISO  (got '{r7.date}')")
_assert(
    r7.specimen_type == "stool", f"specimen_type == 'stool'  (got '{r7.specimen_type}')"
)

# ---------------------------------------------------------------------------
# Test — Flexible specimen detection (alternate formats)
# ---------------------------------------------------------------------------
REPORT_SPECIMEN_FLEX1 = """
URINE CULTURE
Date: 2026-05-01
Organism: E. coli
CFU/mL: 80,000
"""

print("\n=== Test: Flexible Specimen Detection (Urine Culture title) ===")
r8 = extract_structured_data(REPORT_SPECIMEN_FLEX1)
_assert(
    r8.specimen_type == "urine",
    f"specimen_type detected as 'urine' from title  (got '{r8.specimen_type}')",
)

REPORT_SPECIMEN_FLEX2 = """
Specimen Type: Stool
Date: 2026-05-10
Organism: mixed flora
CFU/mL: 2,000
"""

print("\n=== Test: Flexible Specimen Detection (Specimen Type: Stool) ===")
r9 = extract_structured_data(REPORT_SPECIMEN_FLEX2)
_assert(
    r9.specimen_type == "stool",
    f"specimen_type detected as 'stool'  (got '{r9.specimen_type}')",
)

# ---------------------------------------------------------------------------
# Test — Flexible organism detection (alternate formats)
# ---------------------------------------------------------------------------
REPORT_ORG_FLEX1 = """
Specimen: Urine
Date: 2026-06-01
ORGANISM: Klebsiella pneumoniae
CFU/mL: 50,000
"""

print("\n=== Test: Flexible Organism Detection (ORGANISM: caps) ===")
r10 = extract_structured_data(REPORT_ORG_FLEX1)
_assert(
    r10.organism == "Klebsiella pneumoniae",
    f"organism detected from ORGANISM:  (got '{r10.organism}')",
)

REPORT_ORG_FLEX2 = """
Specimen: Urine
Date: 2026-06-15
Isolated: E. coli
CFU/mL: 150,000
"""

print("\n=== Test: Flexible Organism Detection (Isolated:) ===")
r11 = extract_structured_data(REPORT_ORG_FLEX2)
_assert(
    r11.organism == "Escherichia coli",
    f"organism detected from Isolated:  (got '{r11.organism}')",
)

# ---------------------------------------------------------------------------
# Test — Flexible CFU detection (alternate formats)
# ---------------------------------------------------------------------------
REPORT_CFU_FLEX1 = """
Specimen: Urine
Date: 2026-07-01
Organism: E. coli
Result: >100,000 CFU/mL
"""

print("\n=== Test: Flexible CFU Detection (>100,000 format) ===")
r12 = extract_structured_data(REPORT_CFU_FLEX1)
_assert(
    r12.cfu == 100000,
    f"cfu parsed from >100,000 format  (got {r12.cfu})",
)

REPORT_CFU_FLEX2 = """
Specimen: Urine
Date: 2026-07-15
Organism: Enterococcus faecalis
Count: 75,000 colonies per mL
"""

print("\n=== Test: Flexible CFU Detection (Count: + colonies) ===")
r13 = extract_structured_data(REPORT_CFU_FLEX2)
_assert(
    r13.cfu == 75000,
    f"cfu parsed from Count: format  (got {r13.cfu})",
)

# ---------------------------------------------------------------------------
# Test — Flexible date detection (alternate formats)
# ---------------------------------------------------------------------------
REPORT_DATE_FLEX1 = """
Specimen: Urine
Collection Date: 03/25/2026
Organism: E. coli
CFU/mL: 100,000
"""

print("\n=== Test: Flexible Date Detection (Collection Date MM/DD/YYYY) ===")
r14 = extract_structured_data(REPORT_DATE_FLEX1)
_assert(
    r14.date == "2026-03-25",
    f"date parsed from Collection Date:  (got '{r14.date}')",
)

REPORT_DATE_FLEX2 = """
Specimen: Urine
Date: 07-04-2026
Organism: E. coli
CFU/mL: 100,000
"""

print("\n=== Test: Flexible Date Detection (MM-DD-YYYY format) ===")
r15 = extract_structured_data(REPORT_DATE_FLEX2)
_assert(
    r15.date == "2026-07-04",
    f"date parsed from MM-DD-YYYY format  (got '{r15.date}')",
)

# ---------------------------------------------------------------------------
# Test — Keyword-based specimen detection (no explicit Specimen: line)
# ---------------------------------------------------------------------------
REPORT_KEYWORD_URINE = """
URINE CULTURE REPORT
Patient: John Doe
Date: 2026-08-01

MICROBIOLOGY RESULTS:
E. coli isolated at 100,000 CFU/mL
"""

print("\n=== Test: Keyword Specimen Detection (URINE CULTURE) ===")
r16 = extract_structured_data(REPORT_KEYWORD_URINE)
_assert(
    r16.specimen_type == "urine",
    f"specimen_type detected via urine keyword  (got '{r16.specimen_type}')",
)

REPORT_KEYWORD_STOOL = """
FECAL CULTURE
Patient: Jane Smith
Date: 2026-08-15

Salmonella detected
CFU/mL: 45,000
"""

print("\n=== Test: Keyword Specimen Detection (FECAL CULTURE) ===")
try:
    r17 = extract_structured_data(REPORT_KEYWORD_STOOL)
    _assert(
        r17.specimen_type == "stool",
        f"specimen_type detected via fecal keyword  (got '{r17.specimen_type}')",
    )
    _assert(
        r17.cfu == 45000,
        f"cfu == 45000  (got {r17.cfu})",
    )
except ExtractionError as e:
    _assert(False, f"Extraction failed for stool culture test: {e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 50}")
print(f"Extraction Tests Complete: {_PASS} passed, {_FAIL} failed")
if _FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {_FAIL} test(s) failed — review extraction logic")
