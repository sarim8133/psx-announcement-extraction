import os
import re
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, ValidationError
import time
from pathlib import Path
from google import genai
from google.genai import types as genai_types

# Load .env from the project root so GOOGLE_API_KEY is available without
# passing it on the command line.
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# 1. Pydantic Models for Local Strict Validation
# ---------------------------------------------------------------------------
class MeetingDetails(BaseModel):
    date: Optional[str] = None
    time: Optional[str] = None

# Canonical vocabularies. Kept for the prompt + JSON spec and for grading, but
# NOT enforced as Literal types: the harness measures vocabulary drift as a
# field-level mismatch rather than voiding the whole document on a stray value.
ALLOWED_SIGNALS = [
    "Board Meeting", "Closed Period", "Book Closure", "Dividend",
    "Bonus Issue", "Right Issue", "General Meeting", "Director Election",
    "Profit Payment", "Other",
]
ALLOWED_CLOSURE_TYPES = ["Closed Period", "Book Closure"]

class ClosureDetails(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    type: Optional[str] = None  # canonical: ALLOWED_CLOSURE_TYPES

class Financials(BaseModel):
    payout_amount: Optional[float] = None
    currency: Optional[str] = None
    payment_due_date: Optional[str] = None
    entitlement_record_date: Optional[str] = None

class PSXAnnouncement(BaseModel):
    listing_date: str
    document_date: Optional[str] = None
    company_name: str
    subject: str
    announcement_signals: List[str]  # canonical: ALLOWED_SIGNALS
    is_actionable_signal: bool
    # Optional so the model can signal "none" by nulling a whole block; the grader
    # then compares it as all-null fields instead of the schema voiding the document.
    meeting_details: Optional[MeetingDetails] = None
    closure_details: Optional[ClosureDetails] = None
    financials: Optional[Financials] = None

# ---------------------------------------------------------------------------
# 2. Hardcoded Gold Set Oracle (Strict Reference Standards)
# ---------------------------------------------------------------------------
GOLD_SET_ORACLE: Dict[str, Any] = {
    "listing_date": "2026-06-15",
    "document_date": "2026-06-15",
    "company_name": "Atlas Battery Limited",
    "subject": "HOLDING OF BOARD MEETING",
    "announcement_signals": ["Board Meeting", "Closed Period"],
    "is_actionable_signal": True,
    "meeting_details": {
        "date": "2026-06-23",
        "time": "02:30 PM"
    },
    "closure_details": {
        "start_date": "2026-06-16",
        "end_date": "2026-06-23",
        "type": "Closed Period"
    },
    "financials": {
        "payout_amount": None,
        "currency": None,
        "payment_due_date": None,
        "entitlement_record_date": None
    }
}

SET_COMPARE_FIELDS = {"announcement_signals"}

# ---------------------------------------------------------------------------
# 3. Text Normalization Layer (Neutralizing Cosmetic Variance)
# ---------------------------------------------------------------------------
# Day-first ordering for ambiguous numerics: PSX documents are Pakistani (DD/MM/YYYY).
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d",
    "%d-%m-%Y", "%d/%m/%Y",
    "%d %B %Y", "%d %b %Y",
    "%B %d, %Y", "%B %d %Y",
    "%b %d, %Y", "%b %d %Y",
)

def _canonicalize_date(text: str) -> str:
    """Coerce common date renderings to ISO YYYY-MM-DD.

    Unparseable input is returned verbatim (not uppercased) so a genuinely
    wrong date still mismatches the gold value instead of being masked.
    """
    raw = text.strip().replace(".", "")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text.strip()

def normalize_value(val: Any, key_path: str) -> Any:
    """
    Standardizes cosmetic variations (dates, times, spacing, suffix additions)
    so that only genuine semantic extraction errors fail our evaluation run.
    """
    if val is None:
        return None

    if isinstance(val, str):
        cleaned = val.strip()

        # Standardize date fields (e.g. "June 23, 2026" / "23/06/2026" -> "2026-06-23")
        if key_path.endswith("date"):
            return _canonicalize_date(cleaned)
        
        # Standardize currency to ISO code (e.g. "Rs.", "Re.", "Rupees" -> "PKR").
        # Unknown currencies (e.g. "USD") are uppercased but preserved as real content.
        if key_path.endswith("currency"):
            c = re.sub(r"[.\s]", "", cleaned.upper())
            if c in {"RS", "RE", "PKR", "RUPEE", "RUPEES", "RUPEE(S)", "PKRS"}:
                return "PKR"
            return c

        # Standardize Time expressions (e.g. "02:30 PM", "2:30 PM", "2:30 p.m.")
        if key_path.endswith("meeting_details.time"):
            # Uniform lowercase and strip inner periods
            t_clean = cleaned.lower().replace(".", "").replace(" ", "")
            # Pattern match standard time blocks
            match = re.match(r"(\d{1,2}):(\d{2})(am|pm)", t_clean)
            if match:
                hours, minutes, period = match.groups()
                # Zero-pad hours
                hours = f"{int(hours):02d}"
                return f"{hours}:{minutes} {period.upper()}"
            return cleaned.upper()
            
        # Standardize Company Names: case + whitespace only. The full legal name
        # is signal, not noise -- dropping "Limited"/"Company" would let a model
        # that truncated the name score as a correct match (a false equality).
        if key_path.endswith("company_name"):
            return re.sub(r"\s+", " ", cleaned.upper()).strip()
            
        # Standardize Subjects
        if key_path.endswith("subject"):
            return re.sub(r"\s+", " ", cleaned.upper()).strip()
            
    return val

# ---------------------------------------------------------------------------
# 4. Unconstrained Model Extraction
# ---------------------------------------------------------------------------
# Experiment 1: few-shot block that reframes announcement_signals as a fixed-menu
# multiple-choice selection (the model otherwise treats it as a free-text summary).
# Toggled by USE_FEWSHOT so baseline vs. experiment can be A/B-tested from one file.
USE_FEWSHOT = False
USE_ACTIONABLE_RULE = False

FEWSHOT_SIGNALS_BLOCK = """
    announcement_signals is a MULTIPLE-CHOICE selection from EXACTLY these ten values --
    never free text, never a sentence, never the subject line:
      [Board Meeting, Closed Period, Book Closure, Dividend, Bonus Issue,
       Right Issue, General Meeting, Director Election, Profit Payment, Other]
    Select EVERY value that applies (often two), but ONLY from this list, and use the
    value verbatim -- never embellish it ("Profit Payment", not "Semi Annual Profit Payment").

    Examples (the WRONG column is prose/embellishment copied from the document -- never do this):
      Dividend notice          -> ["Dividend"]
                                  NOT ["A dividend of Rs 9.22/unit will be paid", "Interim Distribution"]
      Board meeting + closure   -> ["Board Meeting", "Closed Period"]       (multi-tag)
                                  NOT ["HOLDING OF BOARD MEETING", "Annual Budget 2026-27"]
      Book closure for profit   -> ["Book Closure", "Profit Payment"]       (multi-tag)
                                  NOT ["Book Closure", "Semi Annual Profit Payment"]
      EOGM outcome              -> ["General Meeting"]
                                  NOT ["EOGM Held", "Resolutions Not Approved"]
    If nothing in the list fits, use "Other".
"""

# Experiment 2: explicit past-record-date rule for is_actionable_signal.
# The baseline prompt's actionable definition omits the key edge case: a dividend
# notice whose entitlement record date has already passed is NOT actionable.
IS_ACTIONABLE_RULE_BLOCK = """
    - is_actionable_signal ADDITIONAL RULE (dividend/profit-payment docs only):
      if the document announces a dividend or profit payment and the entitlement
      record date printed in the document is on or before the listing_date supplied
      above, set is_actionable_signal to false — the entitlement window is already
      closed and no investor action is possible. Set true only if the record date is
      strictly in the future relative to listing_date.
      This rule does NOT apply to board meetings, right issues, bonus issues, book
      closures, or general meetings — evaluate those on their own forward/backward
      signal without this date check.
"""

def _build_prompt(listing_date: str) -> str:
    return f"""
    Analyze the attached corporate announcement from the Pakistan Stock Exchange (PSX).
    Output a single valid JSON object adhering to the layout below. Do not wrap it in markdown.

    Metadata context: The listing table date is '{listing_date}'

    Classification Rules:
    - is_actionable_signal: true if the document establishes a future or ongoing restriction,
      entitlement registration, or mandatory action (upcoming board meeting, closed period,
      future dividend entitlement, bonus/right allocations). false if it merely records an
      event/resolution/election that already concluded on or before today with no forward obligation.
    - Strict Sourcing: every non-null field must match verbatim text in the document. If an
      amount or date is not explicitly printed, set it null. Do not extrapolate.
    {FEWSHOT_SIGNALS_BLOCK if USE_FEWSHOT else ""}
    {IS_ACTIONABLE_RULE_BLOCK if USE_ACTIONABLE_RULE else ""}
    Target JSON Structure:
    {{
      "listing_date": "YYYY-MM-DD",
      "document_date": "YYYY-MM-DD or null",
      "company_name": "string",
      "subject": "string",
      "announcement_signals": ["Signal1", "Signal2"],
      "is_actionable_signal": true/false,
      "meeting_details": {{ "date": "YYYY-MM-DD or null", "time": "string or null" }},
      "closure_details": {{ "start_date": "...", "end_date": "...", "type": "Closed Period/Book Closure/null" }},
      "financials": {{ "payout_amount": number or null, "currency": string or null, "payment_due_date": "...", "entitlement_record_date": "..." }}
    }}
    """


def extract_with_file(client: genai.Client, uploaded_file, listing_date: str) -> str:
    """Run extraction on a pre-uploaded file (no upload/delete lifecycle here).

    A nanosecond nonce is appended to the prompt so that each call has a unique
    text, preventing Gemini's implicit response cache from replaying a prior
    result when the same file handle is reused across trials.
    Temperature is set explicitly (1.0) — the API default is also ~1.0, but
    hardcoding it makes the intent unambiguous and future-proofs against default changes.
    """
    nonce_prompt = _build_prompt(listing_date) + f"\n# ref:{time.time_ns()}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=genai_types.GenerateContentConfig(temperature=1.0),
        contents=[uploaded_file, nonce_prompt],
    )
    return response.text


def extract_raw_text_payload(pdf_path: str, listing_date: str) -> str:
    """Upload, extract, and delete in one shot (single-doc runner path)."""
    client = genai.Client()
    uploaded_file = client.files.upload(file=pdf_path)
    try:
        return extract_with_file(client, uploaded_file, listing_date)
    finally:
        client.files.delete(name=uploaded_file.name)

# ---------------------------------------------------------------------------
# 5. Parsing & Local Validation
# ---------------------------------------------------------------------------
def parse_and_validate_json(raw: str) -> dict:
    """Extracts JSON with optional markdown code fence stripping and applies schema validation."""
    text = raw.strip()

    # Strips fenced code blocks if the model fails the "no markdown" prompt constraint
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output was not valid JSON: {exc}") from exc

    try:
        return PSXAnnouncement(**data).model_dump()
    except ValidationError as exc:
        raise ValueError(f"Model output failed schema validation: {exc}") from exc


# ---------------------------------------------------------------------------
# 6. Field-by-Field Comparison Engine
# ---------------------------------------------------------------------------
MATCH = "MATCH"
MISMATCH = "MISMATCH"
REVIEW = "REVIEW"

# Free-text fields where an exact mismatch warrants a human look, not a hard fail.
# (Long subject lines rarely match character-for-character; treat as advisory.)
REVIEW_FIELDS = {"subject"}

# Keys present in the gold record that the model does not produce. Any key
# starting with "_" (e.g. "_note") is also treated as a human annotation, not data.
GOLD_META_KEYS = {"pdf"}


def _compare_node(gold: dict, got: Any, prefix: str, results: List[dict]) -> None:
    """Walk a gold record, comparing each leaf against the model's value."""
    got = got if isinstance(got, dict) else {}
    for key, gold_val in gold.items():
        if key in GOLD_META_KEYS or key.startswith("_"):
            continue
        path = f"{prefix}{key}"
        got_val = got.get(key)

        # Recurse into nested objects (meeting_details, closure_details, financials).
        if isinstance(gold_val, dict):
            _compare_node(gold_val, got_val, path + ".", results)
            continue

        # Order-independent comparison for list fields (announcement_signals).
        if key in SET_COMPARE_FIELDS:
            gold_set = set(gold_val or [])
            got_set = set(got_val or [])
            status = MATCH if gold_set == got_set else MISMATCH
            results.append({"path": path, "status": status,
                            "gold": sorted(gold_set), "got": sorted(got_set)})
            continue

        # Scalar comparison, after cosmetic normalization of both sides.
        g = normalize_value(gold_val, path)
        m = normalize_value(got_val, path)
        if g == m:
            status = MATCH
        elif path in REVIEW_FIELDS:
            status = REVIEW
        else:
            status = MISMATCH
        results.append({"path": path, "status": status, "gold": gold_val, "got": got_val})


def compare_announcement(gold: dict, got: dict) -> List[dict]:
    """Compare one model record against its gold record, field by field."""
    results: List[dict] = []
    _compare_node(gold, got, "", results)
    return results


# ---------------------------------------------------------------------------
# 7. Evaluation Runner
# ---------------------------------------------------------------------------
def grade_document(gold_entry: dict, pdf_dir: str) -> Tuple[List[dict], Optional[str]]:
    """Run the model on one gold entry's PDF and compare. Returns (results, error).

    Any failure (missing file, network/SDK error, bad JSON, schema violation) is
    returned as an error string so one bad document never aborts the whole batch.
    """
    pdf_path = os.path.join(pdf_dir, gold_entry["pdf"])
    if not os.path.exists(pdf_path):
        return [], f"PDF not found: {pdf_path}"
    try:
        raw = extract_raw_text_payload(pdf_path, gold_entry["listing_date"])
        model_dict = parse_and_validate_json(raw)
    except ValueError as exc:
        return [], str(exc)
    except Exception as exc:  # noqa: BLE001 - upload/network errors must not kill the run
        return [], f"{type(exc).__name__}: {exc}"
    return compare_announcement(gold_entry, model_dict), None


def grade_with_file(gold_entry: dict, client: genai.Client, uploaded_file) -> Tuple[List[dict], Optional[str]]:
    """Grade using a pre-uploaded file handle — no upload/delete lifecycle here."""
    try:
        raw = extract_with_file(client, uploaded_file, gold_entry["listing_date"])
        model_dict = parse_and_validate_json(raw)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return compare_announcement(gold_entry, model_dict), None


def run_evaluation(gold_path: str = "gold_set.json", pdf_dir: str = ".") -> None:
    """Grade every PDF named in the gold set and print only the problems + a tally."""
    with open(gold_path, "r", encoding="utf-8") as fh:
        gold_entries = json.load(fh)

    total = matched = mismatched = review = errors = 0

    for entry in gold_entries:
        print(f"\n=== {entry['pdf']} | {entry['company_name']} ===")
        results, error = grade_document(entry, pdf_dir)
        if error:
            errors += 1
            print(f"  [ERROR] {error}")
            continue
        problems = 0
        for r in results:
            total += 1
            if r["status"] == MATCH:
                matched += 1
            elif r["status"] == REVIEW:
                review += 1
                problems += 1
                print(f"  [REVIEW]   {r['path']}: gold={r['gold']!r}  got={r['got']!r}")
            else:
                mismatched += 1
                problems += 1
                print(f"  [MISMATCH] {r['path']}: gold={r['gold']!r}  got={r['got']!r}")
        if problems == 0:
            print("  all fields match")

    print("\n" + "-" * 48)
    print(f"Documents with load/parse errors : {errors}")
    print(f"Fields compared                  : {total}")
    print(f"  Match                          : {matched}")
    print(f"  Mismatch (real errors)         : {mismatched}")
    print(f"  Review (subject, advisory)     : {review}")


if __name__ == "__main__":
    run_evaluation()