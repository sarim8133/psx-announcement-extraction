"""Per-document diff of is_actionable_signal: fewshot vs fewshot+rule.

Prints one row per doc showing gold, both model outputs, and whether each
condition matched. Highlights flips (wrong→right) and regressions (right→wrong).
"""
import json
import pipeline as pipe

GOLD = json.load(open("gold_set.json"))
TARGET = "is_actionable_signal"


def run_condition(label, fewshot, rule):
    pipe.USE_FEWSHOT = fewshot
    pipe.USE_ACTIONABLE_RULE = rule
    out = {}
    for entry in GOLD:
        results, err = pipe.grade_document(entry, ".")
        if err:
            short = err[:60].replace("\n", " ")
            out[entry["pdf"]] = ("ERR", f"ERR:{short}")
            print(f"  [ERR] {entry['pdf']}: {short}", flush=True)
            continue
        found = False
        for r in results:
            if r["path"] == TARGET:
                out[entry["pdf"]] = (r["status"], r["got"])
                found = True
        if not found:
            out[entry["pdf"]] = ("MISSING", "MISSING")
            print(f"  [MISSING] {entry['pdf']}: field not in results", flush=True)
    return out


print("Running fewshot condition …", flush=True)
fs = run_condition("fewshot", fewshot=True, rule=False)

print("Running fewshot+rule condition …", flush=True)
rule = run_condition("fewshot_rule", fewshot=True, rule=True)

print()
print(f"{'PDF':22}  {'gold':5}  {'fewshot':7}  {'fewshot_rule':12}  note")
print("-" * 78)

flipped, regressed, still_wrong = [], [], []

for entry in GOLD:
    pdf = entry["pdf"]
    gold_val = entry.get("is_actionable_signal")
    fs_status, fs_got   = fs.get(pdf,   ("?", None))
    ru_status, ru_got   = rule.get(pdf, ("?", None))

    fs_ok  = fs_status  == pipe.MATCH
    ru_ok  = ru_status  == pipe.MATCH

    if not fs_ok and ru_ok:
        note = "FLIP"
        flipped.append(pdf)
    elif fs_ok and not ru_ok:
        note = "REGRESSION !"
        regressed.append(pdf)
    elif not fs_ok and not ru_ok:
        note = "still wrong"
        still_wrong.append(pdf)
    else:
        note = ""

    fs_tag  = "ok" if fs_ok else str(fs_got)
    ru_tag  = "ok" if ru_ok else str(ru_got)

    print(f"{pdf:22}  {str(gold_val):5}  {fs_tag:7}  {ru_tag:12}  {note}")

print()
print(f"Flipped (wrong→right) : {flipped}")
print(f"Regressions (right→wrong): {regressed}")
print(f"Still wrong under rule   : {still_wrong}")
