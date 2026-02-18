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
}

# ---------------------------------------------------------------------------
# Organism alias normalisation lookup table
# Maps common shorthand/abbreviations → canonical organism name.
# Matching is performed case-insensitively against stripped input.
# ---------------------------------------------------------------------------
ORGANISM_ALIASES: dict = {
    # Escherichia coli variants
    "e. coli": "Escherichia coli",
    "e.coli": "Escherichia coli",
    "e coli": "Escherichia coli",
    "escherichia coli": "Escherichia coli",
    # Klebsiella
    "klebsiella": "Klebsiella pneumoniae",
    "klebsiella pneumoniae": "Klebsiella pneumoniae",
    # Staphylococcus
    "staph aureus": "Staphylococcus aureus",
    "staphylococcus aureus": "Staphylococcus aureus",
    "s. aureus": "Staphylococcus aureus",
    "mrsa": "Staphylococcus aureus (MRSA)",
    # Enterococcus
    "enterococcus": "Enterococcus faecalis",
    "enterococcus faecalis": "Enterococcus faecalis",
    "e. faecalis": "Enterococcus faecalis",
    # Pseudomonas
    "pseudomonas": "Pseudomonas aeruginosa",
    "pseudomonas aeruginosa": "Pseudomonas aeruginosa",
    "p. aeruginosa": "Pseudomonas aeruginosa",
    # Proteus
    "proteus": "Proteus mirabilis",
    "proteus mirabilis": "Proteus mirabilis",
    # Contamination terms (kept as-is but included for normalisation completeness)
    "mixed flora": "mixed flora",
    "skin flora": "skin flora",
    "normal flora": "normal flora",
    "commensal": "commensal",
    "mixed growth": "mixed growth",
}


def normalize_organism(raw: str) -> str:
    """
    Normalise a raw organism string to its canonical name.

    Performs case-insensitive lookup against ORGANISM_ALIASES.
    Returns the canonical name if found, otherwise returns the stripped
    title-cased version of the original input.

    Args:
        raw: Raw organism string from extraction layer.

    Returns:
        Canonical organism name string.
    """
    key = raw.strip().lower()
    return ORGANISM_ALIASES.get(key, raw.strip())
