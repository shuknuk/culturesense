"""
CultureSense Temporal Trend Engine (Cell Group D)
Computes TrendResult from a sorted list of CultureReport objects.
"""

from typing import List

from data_models import CultureReport, TrendResult
from rules import RULES, normalize_organism


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_cfu_trend(cfu_values: List[int]) -> str:
    """
    Classify the CFU trajectory from an ordered list of values.

    Labels (priority order):
        "insufficient_data"  — fewer than 2 reports
        "cleared"            — final value ≤ cleared_threshold (overrides all)
        "decreasing"         — all values monotonically decreasing
        "increasing"         — all values monotonically increasing
        "fluctuating"        — any other pattern
    """
    if len(cfu_values) < 2:
        return "insufficient_data"

    # "cleared" overrides all other labels
    if cfu_values[-1] <= RULES["cleared_threshold"]:
        return "cleared"

    strictly_decreasing = all(
        cfu_values[i] > cfu_values[i + 1] for i in range(len(cfu_values) - 1)
    )
    if strictly_decreasing:
        return "decreasing"

    strictly_increasing = all(
        cfu_values[i] < cfu_values[i + 1] for i in range(len(cfu_values) - 1)
    )
    if strictly_increasing:
        return "increasing"

    return "fluctuating"


def _compute_deltas(cfu_values: List[int]) -> List[int]:
    """
    Compute per-interval CFU changes.

    Positive delta = worsening (increasing CFU).
    Negative delta = improving (decreasing CFU).
    """
    return [cfu_values[i + 1] - cfu_values[i] for i in range(len(cfu_values) - 1)]


def _check_persistence(organism_list: List[str]) -> bool:
    """
    Return True if the same organism was isolated across all reports.

    Comparison is performed on normalised (lowercase, stripped) organism names,
    with alias resolution to handle "E. coli" == "Escherichia coli".
    """
    normalised = [normalize_organism(o).strip().lower() for o in organism_list]
    return len(set(normalised)) == 1


def _check_resistance_evolution(reports: List[CultureReport]) -> bool:
    """
    Return True if new resistance markers appear in any report after the first.

    Logic:
        - Baseline = markers in report[0]
        - If any subsequent report contains a marker not in baseline → True
    """
    if len(reports) < 2:
        return False
    baseline = set(reports[0].resistance_markers)
    later_markers: set[str] = set()
    for r in reports[1:]:
        later_markers.update(r.resistance_markers)
    return bool(later_markers - baseline)


def _build_resistance_timeline(reports: List[CultureReport]) -> List[List[str]]:
    """Return per-report resistance marker lists, in report order."""
    return [list(r.resistance_markers) for r in reports]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_trend(reports: List[CultureReport]) -> TrendResult:
    """
    Compute a TrendResult from an ordered list of CultureReport objects.

    Reports should be sorted by date (oldest first) before calling this
    function. The function does NOT re-sort — caller is responsible.

    Args:
        reports: 1–3 CultureReport instances in chronological order.

    Returns:
        TrendResult with all temporal signal fields populated.
    """
    if not reports:
        raise ValueError("analyze_trend requires at least one CultureReport.")

    cfu_values = [r.cfu for r in reports]
    cfu_deltas = _compute_deltas(cfu_values)
    cfu_trend = _classify_cfu_trend(cfu_values)
    organism_list = [r.organism for r in reports]
    organism_persistent = _check_persistence(organism_list)
    resistance_evolution = _check_resistance_evolution(reports)
    resistance_timeline = _build_resistance_timeline(reports)
    report_dates = [r.date for r in reports]
    any_contamination = any(r.contamination_flag for r in reports)

    return TrendResult(
        cfu_trend=cfu_trend,
        cfu_values=cfu_values,
        cfu_deltas=cfu_deltas,
        organism_persistent=organism_persistent,
        organism_list=organism_list,
        resistance_evolution=resistance_evolution,
        resistance_timeline=resistance_timeline,
        report_dates=report_dates,
        any_contamination=any_contamination,
    )
