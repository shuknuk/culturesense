"""
CultureSense Trend Engine — Unit Tests (Cell D-2)
Kaggle-native inline tests (no pytest dependency).
"""

from data_models import CultureReport
from trend import analyze_trend

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


def _make_report(
    cfu: int,
    organism: str = "Escherichia coli",
    date: str = "2026-01-01",
    markers=None,
    contamination: bool = False,
) -> CultureReport:
    return CultureReport(
        date=date,
        organism=organism,
        cfu=cfu,
        resistance_markers=markers or [],
        susceptibility_profile=[],
        specimen_type="urine",
        contamination_flag=contamination,
        raw_text="<stub>",
    )


# ---------------------------------------------------------------------------
# 1. Monotonically decreasing
# ---------------------------------------------------------------------------
print("=== Test: Monotonically Decreasing CFU ===")
rpts = [
    _make_report(120000, date="2026-01-01"),
    _make_report(40000, date="2026-01-10"),
    _make_report(5000, date="2026-01-20"),
]
t = analyze_trend(rpts)
_assert(t.cfu_trend == "decreasing", f"trend == 'decreasing'  (got '{t.cfu_trend}')")
_assert(t.cfu_deltas == [-80000, -35000], f"deltas correct  (got {t.cfu_deltas})")
_assert(t.organism_persistent is True, f"organism_persistent == True")
_assert(t.resistance_evolution is False, f"resistance_evolution == False")
_assert(t.any_contamination is False, f"any_contamination == False")

# ---------------------------------------------------------------------------
# 2. Cleared (final CFU ≤ 1000) — overrides decreasing
# ---------------------------------------------------------------------------
print("\n=== Test: Cleared (Final CFU ≤ 1000) ===")
rpts2 = [
    _make_report(120000, date="2026-01-01"),
    _make_report(40000, date="2026-01-10"),
    _make_report(800, date="2026-01-20"),
]
t2 = analyze_trend(rpts2)
_assert(t2.cfu_trend == "cleared", f"trend == 'cleared'  (got '{t2.cfu_trend}')")

# ---------------------------------------------------------------------------
# 3. CFU = 0 (no growth) → also cleared
# ---------------------------------------------------------------------------
print("\n=== Test: Zero CFU (No Growth) ===")
rpts3 = [
    _make_report(80000, date="2026-01-01"),
    _make_report(0, date="2026-01-10"),
]
t3 = analyze_trend(rpts3)
_assert(
    t3.cfu_trend == "cleared", f"trend == 'cleared' for CFU=0  (got '{t3.cfu_trend}')"
)

# ---------------------------------------------------------------------------
# 4. Monotonically increasing
# ---------------------------------------------------------------------------
print("\n=== Test: Monotonically Increasing CFU ===")
rpts4 = [
    _make_report(40000, date="2026-01-01"),
    _make_report(80000, date="2026-01-10"),
    _make_report(120000, date="2026-01-20"),
]
t4 = analyze_trend(rpts4)
_assert(t4.cfu_trend == "increasing", f"trend == 'increasing'  (got '{t4.cfu_trend}')")

# ---------------------------------------------------------------------------
# 5. Fluctuating
# ---------------------------------------------------------------------------
print("\n=== Test: Fluctuating CFU ===")
rpts5 = [
    _make_report(80000, date="2026-01-01"),
    _make_report(120000, date="2026-01-10"),
    _make_report(60000, date="2026-01-20"),
]
t5 = analyze_trend(rpts5)
_assert(
    t5.cfu_trend == "fluctuating", f"trend == 'fluctuating'  (got '{t5.cfu_trend}')"
)

# ---------------------------------------------------------------------------
# 6. Single report — insufficient_data
# ---------------------------------------------------------------------------
print("\n=== Test: Single Report (Insufficient Data) ===")
rpts6 = [_make_report(100000, date="2026-01-01")]
t6 = analyze_trend(rpts6)
_assert(
    t6.cfu_trend == "insufficient_data",
    f"trend == 'insufficient_data'  (got '{t6.cfu_trend}')",
)
_assert(t6.cfu_deltas == [], f"deltas == []  (got {t6.cfu_deltas})")

# ---------------------------------------------------------------------------
# 7. Resistance evolution detection
# ---------------------------------------------------------------------------
print("\n=== Test: Resistance Evolution ===")
rpts7 = [
    _make_report(90000, date="2026-01-01", markers=[]),
    _make_report(80000, date="2026-01-10", markers=[]),
    _make_report(75000, date="2026-01-20", markers=["ESBL"]),
]
t7 = analyze_trend(rpts7)
_assert(t7.resistance_evolution is True, f"resistance_evolution == True")
_assert(t7.resistance_timeline[2] == ["ESBL"], f"resistance_timeline[2] == ['ESBL']")

# ---------------------------------------------------------------------------
# 8. Organism change (not persistent)
# ---------------------------------------------------------------------------
print("\n=== Test: Organism Change ===")
rpts8 = [
    _make_report(100000, organism="Escherichia coli", date="2026-01-01"),
    _make_report(90000, organism="Klebsiella pneumoniae", date="2026-01-10"),
]
t8 = analyze_trend(rpts8)
_assert(
    t8.organism_persistent is False,
    f"organism_persistent == False when organism changes",
)

# ---------------------------------------------------------------------------
# 9. Contamination flag propagation
# ---------------------------------------------------------------------------
print("\n=== Test: Contamination Propagation ===")
rpts9 = [
    _make_report(5000, organism="mixed flora", date="2026-01-01", contamination=True),
    _make_report(3000, organism="mixed flora", date="2026-01-10", contamination=True),
]
t9 = analyze_trend(rpts9)
_assert(t9.any_contamination is True, f"any_contamination == True")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 50}")
print(f"Trend Tests Complete: {_PASS} passed, {_FAIL} failed")
if _FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {_FAIL} test(s) failed")
