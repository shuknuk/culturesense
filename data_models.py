"""
CultureSense Data Models
Typed Python dataclasses for structured data flow
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class AntibioticSusceptibility:
    """
    Individual antibiotic susceptibility result from culture report.

    Fields:
        antibiotic: Name of antimicrobial agent (e.g., "Ciprofloxacin")
        mic: Minimum Inhibitory Concentration value (e.g., "<= 0.25", ">= 32")
        interpretation: S/I/R result ("Sensitive", "Intermediate", "Resistant")
        breakpoints: Susceptibility breakpoints (e.g., "<= 0.25 / >= 1")
        notes: Optional clinical notes about this antibiotic
    """

    antibiotic: str
    mic: str
    interpretation: str  # "Sensitive", "Intermediate", "Resistant"
    breakpoints: str = ""
    notes: str = ""


@dataclass
class CultureReport:
    """
    Structured representation of a single culture lab report.

    Fields:
        date: ISO 8601 formatted date string (YYYY-MM-DD)
        organism: Name of identified organism (e.g., "E. coli")
        cfu: Colony Forming Units per mL
        resistance_markers: List of resistance markers (subset of ["ESBL","CRE","MRSA","VRE","CRKP"])
        susceptibility_profile: Full antimicrobial susceptibility table
        specimen_type: Type of specimen ("urine" | "stool" | "unknown")
        contamination_flag: True if organism matches contamination terms
        raw_text: Original report string (NEVER passed to LLM)
    """

    date: str
    organism: str
    cfu: int
    resistance_markers: List[str]
    susceptibility_profile: List[AntibioticSusceptibility]
    specimen_type: str
    contamination_flag: bool
    raw_text: str


@dataclass
class TrendResult:
    """
    Temporal comparison analysis across multiple culture reports.

    Fields:
        cfu_trend: "decreasing" | "increasing" | "fluctuating" | "cleared" | "insufficient_data"
        cfu_values: Ordered list of CFU values across reports
        cfu_deltas: Per-interval changes in CFU
        organism_persistent: True if same organism across all reports
        organism_list: Organism name per report
        resistance_evolution: True if new markers appear in later reports
        resistance_timeline: Resistance markers per report
        report_dates: ISO dates in sorted order
        any_contamination: True if any report flagged as contamination
        multi_drug_resistance: True if any report has 2+ resistance markers
        recurrent_organism_30d: True if same organism recurs within 30 days
        susceptibility_evolution: True if any antibiotic shows S→I, S→R, or I→R transition
        evolved_antibiotics: List of antibiotics that evolved resistance
    """

    cfu_trend: str
    cfu_values: List[int]
    cfu_deltas: List[int]
    organism_persistent: bool
    organism_list: List[str]
    resistance_evolution: bool
    resistance_timeline: List[List[str]]
    report_dates: List[str]
    any_contamination: bool
    multi_drug_resistance: bool = False
    recurrent_organism_30d: bool = False
    susceptibility_evolution: bool = False
    evolved_antibiotics: List[str] = field(default_factory=list)


@dataclass
class HypothesisResult:
    """
    Rule-generated hypothesis with confidence scoring.

    Fields:
        interpretation: Natural language pattern summary (rule-generated)
        confidence: Confidence score [0.0, 0.95] - never 1.0
        risk_flags: List of risk flags (e.g., ["EMERGING_RESISTANCE", "CONTAMINATION"])
        stewardship_alert: True if resistance_evolution is True
        requires_clinician_review: Always True - structural safety guarantee
    """

    interpretation: str
    confidence: float
    risk_flags: List[str]
    stewardship_alert: bool
    requires_clinician_review: bool = True


@dataclass
class MedGemmaPayload:
    """
    Structured payload for MedGemma model inference.

    CRITICAL: raw_text from CultureReport is NEVER included in this payload.
    Only derived structured fields are forwarded.

    Fields:
        mode: "patient" | "clinician"
        trend_summary: Serialized TrendResult
        hypothesis_summary: Serialized HypothesisResult
        safety_constraints: Injected safety instructions
        output_schema: Expected output fields for this mode
    """

    mode: str
    trend_summary: dict
    hypothesis_summary: dict
    safety_constraints: List[str]
    output_schema: dict


@dataclass
class FormattedOutput:
    """
    Final rendered output for either Patient or Clinician mode.

    Fields are mode-specific. Patient mode uses patient_* fields,
    Clinician mode uses clinician_* fields.
    """

    mode: str

    # Patient mode fields
    patient_explanation: Optional[str] = None
    patient_trend_phrase: Optional[str] = None
    patient_questions: Optional[List[str]] = None
    patient_disclaimer: str = ""

    # Clinician mode fields
    clinician_trajectory: Optional[dict] = None
    clinician_interpretation: Optional[str] = None
    clinician_confidence: Optional[float] = None
    clinician_resistance_detail: Optional[str] = None
    clinician_resistance_heatmap: Optional[str] = None
    clinician_stewardship_flag: Optional[bool] = None
    clinician_susceptibility_detail: Optional[str] = None
    clinician_disclaimer: str = ""
