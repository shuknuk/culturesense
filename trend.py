"""
CultureSense Temporal Trend Engine (Cell Group D)
Computes TrendResult from a sorted list of CultureReport objects.
"""

from typing import List

from data_models import CultureReport, TrendResult
from rules import RULES, ORGANISM_ALIASES, ANTIBIOTIC_CLASSES


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


def check_persistence(organism_list: List[str]) -> bool:
    normalized = [
        ORGANISM_ALIASES.get(o.strip().lower(), o.strip().lower())
        for o in organism_list
    ]
    return len(set(normalized)) == 1


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


def _check_susceptibility_evolution(reports: List[CultureReport]) -> tuple:
    """
    Detect S→I, S→R, or I→R transitions for the same antibiotic.

    Only flags evolution if the FINAL report shows worsened susceptibility
    compared to baseline. Transient changes that later resolved do NOT count
    as evolution - we care about the current state.

    Returns:
        (has_evolution, evolved_antibiotics)
        - has_evolution: True if final report shows worsened susceptibility vs baseline
        - evolved_antibiotics: List of antibiotics with ongoing worsened susceptibility
    """
    if len(reports) < 2:
        return False, []

    def normalize_interp(interp: str) -> str:
        """Normalize interpretation to single letter: S, I, or R."""
        upper = interp.strip().upper()
        if upper in ("S", "SENSITIVE", "SUSCEPTIBLE"):
            return "S"
        elif upper in ("I", "INTERMEDIATE"):
            return "I"
        elif upper in ("R", "RESISTANT"):
            return "R"
        return upper

    # Build baseline from first report
    baseline: dict = {}  # antibiotic -> interpretation
    for susc in reports[0].susceptibility_profile:
        abx = susc.antibiotic.strip().lower()
        baseline[abx] = normalize_interp(susc.interpretation)

    # Build final state from LAST report
    final_state: dict = {}  # antibiotic -> interpretation
    for susc in reports[-1].susceptibility_profile:
        abx = susc.antibiotic.strip().lower()
        final_state[abx] = normalize_interp(susc.interpretation)

    # Check if final state shows worsening vs baseline
    evolved = []
    for abx, final_interp in final_state.items():
        if abx in baseline:
            baseline_interp = baseline[abx]
            # Detect worsening: S→I, S→R, I→R (compare final vs baseline)
            if (baseline_interp == "S" and final_interp in ("I", "R")) or \
               (baseline_interp == "I" and final_interp == "R"):
                # Find original case from reports for display
                for susc in reports[-1].susceptibility_profile:
                    if susc.antibiotic.strip().lower() == abx:
                        evolved.append(susc.antibiotic.strip())
                        break

    return len(evolved) > 0, evolved


def _check_multi_drug_resistance(reports: List[CultureReport]) -> bool:
    """
    Return True if any single report shows resistance to >= 2 antibiotic classes.

    Multi-drug resistance (MDR) is defined as resistance to >= 2 distinct
    antibiotic classes (not just 2 individual antibiotics). This function:
        1. Checks high-risk resistance markers (ESBL, CRE, MRSA, VRE, CRKP)
        2. Counts distinct antibiotic classes with resistance from susceptibility profile

    Returns True if either condition indicates MDR pattern.
    """
    # First check: high-risk markers always trigger MDR flag
    high_risk_markers = set(RULES.get("high_risk_markers", []))
    for r in reports:
        if any(marker in high_risk_markers for marker in r.resistance_markers):
            return True

    # Second check: count distinct antibiotic classes with resistance
    # MDR = resistance to >= 2 distinct classes
    threshold = RULES.get("multi_drug_threshold", 2)

    for r in reports:
        resistant_classes = set()

        for susc in r.susceptibility_profile:
            # Normalize antibiotic name to lookup key
            abx_key = susc.antibiotic.strip().lower()

            # Check if this antibiotic shows resistance (handles "R" or "Resistant")
            interp = susc.interpretation.upper()
            if interp == "R" or interp == "RESISTANT":
                # Map to antibiotic class
                abx_class = ANTIBIOTIC_CLASSES.get(abx_key)
                if abx_class:
                    resistant_classes.add(abx_class)

        # MDR if resistant to >= threshold distinct classes
        if len(resistant_classes) >= threshold:
            return True

    return False


def _build_resistance_timeline(reports: List[CultureReport]) -> List[List[str]]:
    """Return per-report resistance marker lists, in report order."""
    return [list(r.resistance_markers) for r in reports]


def _check_recurrent_organism(reports: List[CultureReport]) -> bool:
    """
    Return True if the same organism recurs after apparent resolution.

    Recurrence means:
        1. A prior report showed cleared/no growth (CFU ≤ cleared_threshold), AND
        2. The same organism reappears in a later report within 30 days

    Sequential monitoring of the same infection (same organism across reports
    without clearing) is NOT recurrence - it's treatment tracking.

    This is important for stewardship alerts: we only want to flag true
    relapse/recurrence scenarios, not normal treatment monitoring.
    """
    if len(reports) < 2:
        return False

    # Get reports with valid dates, including CFU for resolution check
    from datetime import datetime, timedelta

    dated_reports = []
    for r in reports:
        if r.date and r.date not in ("unknown", ""):
            try:
                date_obj = datetime.strptime(r.date, "%Y-%m-%d")
                normalized_org = ORGANISM_ALIASES.get(
                    r.organism.strip().lower(), r.organism.strip().lower()
                )
                dated_reports.append((date_obj, normalized_org, r.cfu))
            except (ValueError, AttributeError):
                continue

    if len(dated_reports) < 2:
        return False

    # Sort by date
    dated_reports.sort(key=lambda x: x[0])

    # Check for recurrence: cleared → same organism reappears
    cleared_threshold = RULES.get("cleared_threshold", 1000)

    for i in range(len(dated_reports)):
        date_i, org_i, cfu_i = dated_reports[i]

        # Check if this report showed resolution
        is_resolved = cfu_i <= cleared_threshold

        if is_resolved:
            # Check if same organism appears again later
            for j in range(i + 1, len(dated_reports)):
                date_j, org_j, cfu_j = dated_reports[j]

                # Recurrence: cleared → same organism reappears (above threshold)
                if org_i == org_j and cfu_j > cleared_threshold:
                    if (date_j - date_i) <= timedelta(days=30):
                        return True

    return False


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
    organism_persistent = check_persistence(organism_list)
    resistance_evolution = _check_resistance_evolution(reports)
    resistance_timeline = _build_resistance_timeline(reports)
    report_dates = [r.date for r in reports]

    any_contamination = any(r.contamination_flag for r in reports)
    multi_drug_resistance = _check_multi_drug_resistance(reports)
    recurrent_organism_30d = _check_recurrent_organism(reports)

    # Check for susceptibility evolution (S→I, S→R, I→R transitions)
    susc_evolution, evolved_antibiotics = _check_susceptibility_evolution(reports)

    # Combined resistance evolution: either high-risk markers or susceptibility changes
    combined_resistance_evolution = resistance_evolution or susc_evolution

    return TrendResult(
        cfu_trend=cfu_trend,
        cfu_values=cfu_values,
        cfu_deltas=cfu_deltas,
        organism_persistent=organism_persistent,
        organism_list=organism_list,
        resistance_evolution=combined_resistance_evolution,
        resistance_timeline=resistance_timeline,
        report_dates=report_dates,
        any_contamination=any_contamination,
        multi_drug_resistance=multi_drug_resistance,
        recurrent_organism_30d=recurrent_organism_30d,
        susceptibility_evolution=susc_evolution,
        evolved_antibiotics=evolved_antibiotics,
    )
