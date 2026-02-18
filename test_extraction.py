"""
CultureSense Extraction Layer — Unit Tests (Cell C-3)
Kaggle-native inline tests (no pytest dependency).
"""

import warnings
from extraction import extract_structured_data, ExtractionError

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
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 50}")
print(f"Extraction Tests Complete: {_PASS} passed, {_FAIL} failed")
if _FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {_FAIL} test(s) failed — review extraction logic")
