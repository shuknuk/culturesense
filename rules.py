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
    # Per CLAUDE.md Section 5.4: stewardship alert fires at 2+ classes
    "multi_drug_threshold": 2,
    "min_confidence": 0.20,
    "confidence_high_base": 0.90,
    "confidence_longitudinal_penalty": 0.20,
    "confidence_symptom_penalty": 0.20,
}

# ---------------------------------------------------------------------------
# Antibiotic class mapping for MDR detection
# Maps individual antibiotics to their drug classes for resistance counting.
# A multi-drug resistant (MDR) organism is defined as resistance to >=2
# distinct antibiotic classes.
# ---------------------------------------------------------------------------
ANTIBIOTIC_CLASSES: dict = {
    # Beta-lactams
    "ampicillin": "beta_lactam",
    "amoxicillin": "beta_lactam",
    "amoxicillin/clavulanate": "beta_lactam",
    "piperacillin/tazobactam": "beta_lactam",
    "cefazolin": "beta_lactam",
    "cefuroxime": "beta_lactam",
    "ceftriaxone": "beta_lactam",
    "ceftazidime": "beta_lactam",
    "cefepime": "beta_lactam",
    "ertapenem": "beta_lactam",
    "meropenem": "beta_lactam",
    "imipenem": "beta_lactam",
    "aztreonam": "beta_lactam",
    "penicillin": "beta_lactam",
    "oxacillin": "beta_lactam",
    "nafcillin": "beta_lactam",
    "ticarcillin/clavulanate": "beta_lactam",

    # Fluoroquinolones
    "ciprofloxacin": "fluoroquinolone",
    "levofloxacin": "fluoroquinolone",
    "moxifloxacin": "fluoroquinolone",
    "ofloxacin": "fluoroquinolone",
    "norfloxacin": "fluoroquinolone",

    # Aminoglycosides
    "gentamicin": "aminoglycoside",
    "tobramycin": "aminoglycoside",
    "amikacin": "aminoglycoside",

    # Sulfonamides
    "trimethoprim/sulfamethoxazole": "sulfonamide",
    "tmp/smx": "sulfonamide",
    "tmp-smx": "sulfonamide",
    "sulfamethoxazole": "sulfonamide",

    # Tetracyclines
    "tetracycline": "tetracycline",
    "doxycycline": "tetracycline",
    "minocycline": "tetracycline",
    "tigecycline": "tetracycline",

    # Nitrofurans
    "nitrofurantoin": "nitrofuran",

    # Glycopeptides
    "vancomycin": "glycopeptide",
    "teicoplanin": "glycopeptide",

    # Lipopeptides
    "daptomycin": "lipopeptide",

    # Oxazolidinones
    "linezolid": "oxazolidinone",

    # Phenicols
    "chloramphenicol": "phenicol",

    # Fosfomycins
    "fosfomycin": "fosfomycin",

    # Macrolides
    "erythromycin": "macrolide",
    "azithromycin": "macrolide",
    "clarithromycin": "macrolide",

    # Lincosamides
    "clindamycin": "lincosamide",

    # Streptogramins
    "quinupristin/dalfopristin": "streptogramin",

    # Polymyxins
    "colistin": "polymyxin",
    "polymyxin b": "polymyxin",
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
    # Stool pathogens
    "salmonella": "salmonella",
    "shigella": "shigella",
    "campylobacter": "campylobacter",
    "c. diff": "clostridioides difficile",
    "c.diff": "clostridioides difficile",
    "clostridioides difficile": "clostridioides difficile",
    "clostridium difficile": "clostridioides difficile",
    "giardia": "giardia lamblia",
    "cryptosporidium": "cryptosporidium",
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
