"""
Heuristic document-class inference from extracted text + filename.
Tenant referencing focus: immigration, medical, and income/banking evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

VALID_KEYS = ("visa_refusal", "medical_letter", "income_evidence")

# (keyword, weight) — longer phrases get higher weight when matched
VISA_PATTERNS: List[Tuple[str, float]] = [
    ("ukvi", 4.0),
    ("home office", 4.0),
    ("entry clearance", 4.0),
    ("leave to remain", 4.0),
    ("immigration", 3.0),
    ("visa refusal", 5.0),
    ("refusal letter", 4.0),
    ("appendix fm", 3.0),
    ("appendix", 1.5),
    ("eea family permit", 4.0),
    ("settlement", 2.5),
    ("asylum", 2.5),
    ("border force", 3.0),
    ("points-based", 3.0),
]

MEDICAL_PATTERNS: List[Tuple[str, float]] = [
    ("gp letter", 4.0),
    ("general practitioner", 3.0),
    ("nhs", 3.0),
    ("medical certificate", 4.0),
    ("fitness to work", 3.5),
    ("sick note", 3.5),
    ("diagnosis", 2.5),
    ("patient name", 2.0),
    ("clinical", 2.5),
    ("hospital", 2.0),
    ("prescription", 2.0),
    ("doctor", 1.5),
    ("physician", 2.0),
    ("surgery", 1.5),
    ("health centre", 2.5),
]

# Payslips, bank statements, payroll (tenant referencing income checks)
INCOME_PATTERNS: List[Tuple[str, float]] = [
    ("payslip", 5.0),
    ("pay slip", 4.0),
    ("payroll", 4.0),
    ("gross pay", 4.0),
    ("net pay", 4.0),
    ("paye", 3.5),
    ("employer", 2.5),
    ("national insurance", 3.5),
    ("ni number", 3.0),
    ("hmrc", 3.5),
    ("salary", 3.0),
    ("wage", 2.5),
    ("bank statement", 5.0),
    ("sort code", 4.0),
    ("account number", 3.5),
    ("current account", 3.5),
    ("balance brought forward", 3.5),
    ("direct debit", 3.0),
    ("standing order", 3.0),
    ("credit", 1.5),
    ("debit", 1.5),
    ("barclays", 2.0),
    ("lloyds", 2.0),
    ("hsbc", 2.0),
    ("nationwide", 2.0),
    ("santander", 2.0),
    ("natwest", 2.0),
    ("monzo", 2.0),
    ("starling", 2.0),
]


def _score_patterns(blob: str, patterns: List[Tuple[str, float]]) -> float:
    total = 0.0
    for phrase, w in patterns:
        if phrase in blob:
            total += w
    return total


def infer_doc_type_key(analysis: Dict[str, Any], filename: str) -> Tuple[str, str, str]:
    """
    Returns (doc_type_key, confidence, short_reason).

    confidence: "high" | "medium" | "low"
    """
    raw = (analysis.get("metadata") or {}).get("raw_data") or {}
    text = raw.get("extracted_text_full") or raw.get("extracted_text") or ""
    if isinstance(text, str):
        text_full = text[:80000]
    else:
        text_full = ""
    blob = text_full.lower()
    fn = (filename or "").lower()
    blob = f"{blob}\n{fn}"

    assessment = raw.get("assessment")
    if isinstance(assessment, str) and assessment.strip():
        blob += "\n" + assessment.lower()

    inst = raw.get("institutional_indicators")
    if isinstance(inst, list):
        blob += "\n" + " ".join(str(x).lower() for x in inst)
    elif isinstance(inst, str):
        blob += "\n" + inst.lower()

    scores: Dict[str, float] = {
        "visa_refusal": _score_patterns(blob, VISA_PATTERNS),
        "medical_letter": _score_patterns(blob, MEDICAL_PATTERNS),
        "income_evidence": _score_patterns(blob, INCOME_PATTERNS),
    }

    # Filename-only nudges (referencing workflows)
    if "medical" in fn or "gp" in fn or "doctor" in fn:
        scores["medical_letter"] += 2.0
    if any(
        x in fn
        for x in (
            "payslip",
            "payroll",
            "salary",
            "bank",
            "statement",
            "barclays",
            "lloyds",
            "hsbc",
        )
    ):
        scores["income_evidence"] += 2.5
    if "refusal" in fn or "ukvi" in fn or "visa" in fn:
        scores["visa_refusal"] += 2.0

    best_key = max(scores, key=lambda k: scores[k])
    best = scores[best_key]
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
    margin = best - second

    if best < 1.5:
        key = "income_evidence"
        confidence = "low"
        reason = (
            "No strong document-type signals in text; defaulted to payslip / bank statement "
            "(common for referencing income checks)."
        )
        return key, confidence, reason

    if margin >= 4.0 and best >= 5.0:
        confidence = "high"
    elif margin >= 2.0 or best >= 6.0:
        confidence = "medium"
    else:
        confidence = "low"

    label_hint = {
        "visa_refusal": "immigration / refusal cues",
        "medical_letter": "medical / clinical cues",
        "income_evidence": "payslip / payroll / banking cues",
    }[best_key]
    reason = f"Matched {label_hint} in extracted content (score {best:.1f})."

    return best_key, confidence, reason
