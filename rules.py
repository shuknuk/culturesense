"""
CultureSense Rule Library
Clinical constants, thresholds, and organism normalization tables.
"""

# ---------------------------------------------------------------------------
# Core clinical rules and thresholds
# ---------------------------------------------------------------------------
RULES = {
    # CFU/mL threshold above which a urine specimen is considered infected
    "infection_threshold_urine": 100000,
    # CFU/mL threshold above which a stool specimen is considered infected
    "infection_threshold_stool": 50000,
    # A reduction of 75%+ from the previous reading is a strong improvement
    "significant_reduction_pct": 0.75,
    # Organism names indicating sample contamination rather than true infection
    "contamination_terms": [
        "mixed flora",
        "skin flora",
        "normal flora",
        "commensal",
        "contamination",
        "mixed growth",
    ],
    # High-risk resistance markers tracked by the rule engine
    "high_risk_markers": ["ESBL", "CRE", "MRSA", "VRE", "CRKP"],
    # CFU/mL at or below this value is treated as effectively cleared
    "cleared_threshold": 1000,
    # Hard ceiling on confidence - epistemic humility; never 1.0
    "max_confidence": 0.95,
    # Starting confidence before any signal adjustments
    "base_confidence": 0.50,
    # Number of resistant antibiotics to flag as multi-drug resistance
    "multi_drug_threshold": 3,
}

# ---------------------------------------------------------------------------
# Organism alias normalisation lookup table
# Maps common shorthand/abbreviations → canonical organism name.
# Matching is performed case-insensitively against stripped input.
# ---------------------------------------------------------------------------
ORGANISM_ALIASES: dict = {
    # Escherichia coli variants
    "e. coli": "escherichia coli",
    "e.coli": "escherichia coli",
    "e coli": "escherichia coli",
    "escherichia coli": "escherichia coli",
    # Klebsiella
    "klebsiella": "klebsiella pneumoniae",
    "klebsiella pneumoniae": "klebsiella pneumoniae",
    # Staphylococcus
    "staph aureus": "staphylococcus aureus",
    "staphylococcus aureus": "staphylococcus aureus",
    "s. aureus": "staphylococcus aureus",
    "mrsa": "staphylococcus aureus (mrsa)",
    # Enterococcus
    "enterococcus": "enterococcus faecalis",
    "enterococcus faecalis": "enterococcus faecalis",
    "e. faecalis": "enterococcus faecalis",
    # Pseudomonas
    "pseudomonas": "pseudomonas aeruginosa",
    "pseudomonas aeruginosa": "pseudomonas aeruginosa",
    "p. aeruginosa": "pseudomonas aeruginosa",
    # Proteus
    "proteus": "proteus mirabilis",
    "proteus mirabilis": "proteus mirabilis",
    # Contamination terms (kept as-is but included for normalisation completeness)
    "mixed flora": "mixed flora",
    "skin flora": "mixed flora",
    "normal flora": "mixed flora",
    "commensal": "commensal",
    "mixed growth": "mixed flora",
}


def normalize_organism(raw: str) -> str:
    """
    Normalise a raw organism string to its canonical name.

    Performs case-insensitive lookup against ORGANISM_ALIASES.
    Returns the canonical name if found, otherwise returns the stripped
    original input.

    Args:
        raw: Raw organism string from extraction layer.

    Returns:
        Canonical organism name string.
    """
    key = raw.strip().lower()
    canonical = ORGANISM_ALIASES.get(key, raw.strip())
    # Contamination terms stay lowercase, others get first letter capitalized
    if canonical in ("mixed flora", "skin flora", "normal flora", "commensal"):
        return canonical
    # Capitalize first letter only (e.g., "escherichia coli" -> "Escherichia coli")
    if canonical:
        return canonical[0].upper() + canonical[1:] if len(canonical) > 1 else canonical.upper()
    return raw.strip()
