# PSX Announcement Extraction

**Structured data from scanned financial filings via a vision LLM — built around an evaluation-driven development loop.**

This is a learning project, not a product. The point isn't the extractor; it's the loop around it: a hand-labeled gold set, a field-by-field evaluation harness, and three controlled prompt experiments — each with a stated hypothesis, a measured result reported with variance, and a named residual failure. I built it from raw API calls, no framework, so the mechanics (JSON parsing, schema validation, evaluation) stay visible.

---

## 1. The problem

Pakistan Stock Exchange (PSX) corporate announcements — dividends, book closures, board meetings, index changes — are published as scanned PDFs. The actionable facts are locked inside unstructured documents. No free structured API exists (Capital Stake is paid; the rest are price-only feeds). **That absence is the reason the project is worth doing.**

The goal: extract structured JSON reliably from these documents — and, the real point, **measure how reliably**, then improve it in measured steps.

---

## 2. Architecture

```
 PSX portal          ┌──────────────────────────────┐
 (scanned PDFs) ───► │ Stage 1 — title routing      │  most discarded
                     │ rule-based, NO LLM           │  (director elections,
                     └──────────────┬───────────────┘   routine disclosures)
                                    │ signal-bearing docs
                                    ▼
                     ┌──────────────────────────────┐
                     │ Stage 2 — vision LLM extract │
                     │ Gemini 2.5 Flash, raw API    │
                     └──────────────┬───────────────┘
                                    │ raw JSON string
                                    ▼
                     ┌──────────────────────────────┐
                     │ parse · strip fences ·       │
                     │ Pydantic validate            │
                     └──────────────┬───────────────┘
                                    │ structured record
                                    ▼
                     ┌──────────────────────────────┐
                     │ eval harness — field-by-field│
                     │ diff vs hand-labeled gold set│
                     └──────────────────────────────┘
```

Three decisions worth defending:

- **Vision, not OCR.** The source PDFs are mostly clean scans. The model reads the page image directly — no Tesseract stage. More accurate on messy scans, fewer moving parts, and OCR-wrangling teaches nothing about *LLM* engineering.
- **Two stages.** A cheap rule-based title filter discards the bulk of PSX volume — routine notices like director elections and disclosures — before any LLM call runs (the majority of a given day's listing, estimated from manual inspection rather than measured across the full corpus). Knowing when **not** to call the model is part of the point.
- **Raw API calls, no framework.** I write the parsing, fence-stripping, and validation by hand. I explicitly removed Gemini's `response_schema` auto-enforcement early on because it hid the exact skill I was trying to learn.

---

## 3. The gold set

15 hand-labeled documents. Every non-null field is sourced to verbatim text on the page — **if you can't point to the words, it's null; never extrapolate.** A wrong gold label is worse than no label: it makes the evaluation reward wrong behavior.

The set is **stratified deliberately, not sampled randomly.** Random sampling of PSX volume would have handed me ~15 near-identical board-meeting notices. Instead the set spans Board Meeting (×3), Dividend (×4), Closed Period (×2), General Meeting (×2), Director Election (×2), Book Closure, Profit Payment, Other (×3), and Right Issue (×1).

**Honest gaps:**
- **Bonus Issue** has zero examples — an entirely untested enum value.
- **Right Issue** has one, and it's a *schema-limitation* case: the model can recognize it but the v1 schema can't hold its terms (ratio, subscription price, multi-step schedule).

---

## 4. The evaluation harness

This is the part most junior projects skip entirely, so it gets room.

The harness does a recursive, field-by-field diff of the model's output against the gold set and returns matches and mismatches with exact field paths. Four design decisions carry the weight:

- **Measure, don't enforce.** Out-of-vocabulary values are *recorded as mismatches*, not treated as fatal. I learned this the hard way: a strict Pydantic `Literal` enum was voiding entire documents over one bad signal tag. I changed `announcement_signals` to `List[str]` — the vocabulary stays enforced in the prompt and checked at compare-time, but one bad field no longer nukes the other eleven.
- **Normalize format, never content.** Dates, times (`2:30 PM` ≡ `02:30 PM`), and currency (`Rs./Re./Rupees` → `PKR`) are canonicalized at compare-time so cosmetic noise doesn't masquerade as a semantic error. A *failed* version of this normalized company names too — it deleted words like "Limited" and "Company," which hides real errors. I removed it. Normalization fixes format; it must never touch content.
- **Signals compared as a set.** `announcement_signals` is order-independent — a correct-but-reordered tag list scores as a match.
- **One toggle, true A/B.** Few-shot and rule blocks are flags (`USE_FEWSHOT`, `USE_ACTIONABLE_RULE`) in the same file, so baseline and experiment run from identical code. The baseline is never lost.

Every run is **3 trials at temperature 1.0 with a per-call nonce** (`time.time_ns()` appended to the prompt) to defeat the implicit response cache. I report mean and range, never a bare number.

---

## 5. Results — the experiment arc

| Stage | `signals_ok` /15 | `actionable_ok` /15 | overall % (all 15) | corrected % (14) |
|-------|:---:|:---:|:---:|:---:|
| Baseline | 2.3 `[2, 3]` | — | 85.8 `[84.4, 86.7]` | 88.4 `[87.1, 89.5]` |
| + Exp 1 — few-shot signals | 11.3 `[10, 12]` | 10.0 `[10, 10]` | 88.3 `[86.7, 89.8]` | 90.9 `[89.0, 92.4]` |
| + Exp 2b — narrowed actionable rule | 11.0 `[11, 11]` | 13.0 `[13, 13]` | 91.1 `[90.2, 92.9]` | 93.6 `[93.3, 94.3]` |

> **Two denominators, both reported on purpose.** *Overall %* is every field across all 15 documents. *Corrected %* excludes the one Ghani right-issue document (8 of its fields can't match because the v1 schema has no place for right-issue terms — see §7). I report both so the headline isn't a cherry-pick: the schema-limitation case is real and stays in the all-15 number; the corrected number just isolates "how well does the model do on documents the schema can actually represent."
>
> **On `actionable_ok` at baseline (`—`):** I never measured this field standalone *before* the few-shot prompt existed, so there's no true pre-everything baseline for it. The honest comparison is therefore **Exp 1's 10.0 `[10,10]` → Exp 2b's 13.0 `[13,13]`** — i.e. what the actionable rule added, holding few-shot constant. I'm flagging this rather than back-filling a baseline I didn't run.
>
> `signals_ok` counts documents with the **entire** signal array exactly right — all-or-nothing per document, which is why even a strong model scores low on it.

Each experiment below leads with the **mechanism** — what I thought was wrong and why — then the result with range, then what it still gets wrong.

### Experiment 1 — few-shot contrast examples for `announcement_signals`

**Hypothesis.** The failure wasn't a vocabulary problem; it was a *field-nature* problem. The model was treating a multiple-choice field as a free-text summary — returning whole sentences like `["A dividend of Rs 9.22/unit will be paid"]` where the schema wanted `["Dividend"]`. It didn't know the field's *form*, not its *words*.

**Fix.** Few-shot contrast examples: the correct enum array placed directly next to the wrong prose the model actually produced. Multi-tag examples on both sides to stop it over-collapsing *and* over-splitting.

**Result.** `signals_ok` 2.3 `[2,3]` → 11.3 `[10,12]`. **The ranges don't overlap** — this is a real effect, not sampling noise. Overall match moved only 85.8 → 88.3, which is *expected*: signals is 1 of ~15 fields, and a targeted fix should not move the aggregate much.

**Residual.** Dividend/Profit-Payment conflation on ~2 docs; "Other" misidentification on ~2 more. Genuinely ambiguous cases, not form errors.

### Experiment 2 → 2b — the actionable rule, and the regression

This is the experiment I'd lead an interview with, because the interesting part is a regression the aggregate hid.

**Hypothesis.** `is_actionable_signal` was a *spec gap*, not model bias. The prompt never stated the rule that a dividend whose record date has **already passed** at issuance is not actionable — the entitlement window is closed, no investor action is possible.

**Exp 2 (broad rule).** I added the rule, mentioning dividends, profit payments, *and* right issues. Aggregate: `actionable_ok` 10.7 → 12.3. Net +1 — looks fine. **It wasn't.** The document-level diff told a different story:

- **+2 flips (wrong → right):** 278597 and 278586 — both past-record-date dividends, flipping exactly as hypothesized. Mechanism confirmed at the document level.
- **−1 regression (right → wrong):** 278435, the Ghani **right issue**. Because the rule named "right issue," the model applied past-record-date logic to it, read the rights *record* date, and wrongly concluded the window had closed. Gold = `True` (the subscription and allotment dates are in the future); model = `False`. **Over-broad rule scope.**

A mean going up by 1 concealed a doc that broke. Per-document diffs catch what aggregates hide.

**Exp 2b (narrowed rule).** I scoped the rule to dividends and profit payments only, with explicit exclusions for board meetings, right issues, bonus issues, and book closures. Result: `actionable_ok` 10/15 every trial → **13/15 every trial**, ranges don't touch. The Ghani regression recovered, both dividend flips held, no new regression. `corrected %` climbed 88.4 (baseline) → 90.9 (Exp 1) → **93.6** (Exp 2b).

Three different fix types across the arc — a *format* fix (few-shot), a *logic* fix (the rule), and a *scope* correction (narrowing it) — each with a known mechanism and a quantified result.

---

## 6. The variance finding

Same model, same temperature (1.0, nonce-busted), yet `is_actionable` was stable at **13/13 across every trial** while `signals_ok` varied across `[10, 12]`. That looks contradictory until you look at *where* the stability lives.

The stability is a property of the **gold set's evidence structure, not the model's determinism:**

- A boolean backed by **unambiguous** document evidence concentrates almost all probability mass on one answer — sampling at temp 1.0 doesn't flip it.
- The remaining `is_actionable` failures are **structurally unreachable** (no printed date for the rule to anchor to), so the model is *consistently* wrong on them too — also stable.
- `signals_ok` varies because multi-tag and "Other" calls have **genuinely split** probability mass — real ambiguity, so real run-to-run variance.

The honest framing is "**stable because the reachable cases are unambiguous and the failures are structural — not because the model is deterministic.**" I do not claim zero variance as a model property; it's a task property, and conflating the two would be a false claim about sampling.

---

## 7. Limitations & ceilings

Named proactively, because naming your own ceilings reads as confidence.

- **`is_actionable` tops out at 13/15 with this approach.** The two residual misses (278602, 278443) are structurally unreachable by a record-date rule — there is no printed date to anchor to. Breaking past 13/15 needs a *different mechanism* (reasoning over forward/backward language, or a separate prompt path), not more tuning.
- **`Bonus Issue` is defined but unvalidated.** It's in the signal vocabulary, but the gold set has zero bonus-issue documents — so that enum value has never actually been tested against a real example. The code path exists; the evidence for it doesn't. (Filling this is deliberately left to v2 rather than reopening a finished evaluation.)
- **The schema can't hold several real things:** right/bonus-issue terms (ratio, subscription price, multi-step schedule), multi-tranche record dates (single-value field nulls them), Shariah-compliance status (my actual investment filter), and meeting *outcomes* ("resolutions failed") — only meeting *type* is captured.
- **The model audited the gold set.** Running extraction surfaced real labeling errors I'd made — e.g. 278628 (Sapphire), where gold said `document_date: null` but both the model and the subject line ("HELD ON 15TH JUNE 2026") pointed to a real date on the page. The evaluation works in *both* directions: it grades the model *and* the answer key. That's a feature, not an embarrassment.

---

## 8. What I'd do next (v2)

- A dedicated `corporate_action` schema block so right/bonus issues have a home.
- A multi-tranche `entitlement_record_date` array.
- A `shariah_compliance` field — the filter I actually invest by.
- A non-record-date mechanism for the unreachable `is_actionable` cases.

Short list on purpose. I know where it goes without having gold-plated v1.

---

## 9. Tech & how to run

**Stack:** Python · `google-genai` SDK · Gemini 2.5 Flash (vision) · Pydantic (local validation). No LangChain/LangGraph — "from scratch" is the point.

```bash
# 1. provide your key — copy the template and paste your Gemini key
#    (the code reads GOOGLE_API_KEY, the google-genai SDK default; .env is auto-loaded)
cp .env.example .env

# 2. run the A/B experiment harness: 3 trials, 5 parallel workers
python run_experiment.py 3 5

# grade every doc in the gold set, print mismatches + a tally (this produced full_run.txt)
python pipeline.py
```

**Legal / data:** a learning project over public regulatory disclosures. PSX restricts redistribution of its *market-data feed*, which this project doesn't touch. The repo publishes the **code, the hand-labeled gold set, and the run logs** — the evidence behind every number above. The **source PDFs are gitignored**: they're the bulky originals and not mine to redistribute. Don't deploy this as a public real-time feed.

---

*Built as a portfolio project to demonstrate evaluation-driven LLM engineering: I don't just build the extractor — I measure it, find where it breaks, and fix it in attributable steps.*