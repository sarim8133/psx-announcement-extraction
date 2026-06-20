"""Optimized A/B experiment driver for PSX extraction prompt.

Two key optimizations over v1:
  1. Upload each of the 15 PDFs once; reuse the server-side handle across all
     conditions and trials (was: upload+delete on every single API call).
  2. Grade all 15 docs in parallel within each trial via ThreadPoolExecutor
     (was: sequential, ~10-20 s × 15 docs per trial).

Typical runtime: ~4-6 min for 3 trials × 2 conditions (was ~25-30 min).

Usage:  python run_experiment.py [trials [max_workers]]
        Defaults: 3 trials, 5 concurrent workers.
        Lower max_workers to 2-3 if you hit 429 rate-limit errors.
"""
import json
import os
import sys
import statistics
from concurrent.futures import ThreadPoolExecutor

from google import genai
import pipeline as pipe

GOLD = json.load(open("gold_set.json", encoding="utf-8"))
SCHEMA_LIMITED_PDFS = {"278435-1.pdf"}


def upload_all(client: genai.Client, pdf_dir: str = ".") -> dict:
    """Upload all PDFs in parallel. Returns {filename: file_obj | Exception}."""
    def _upload(entry):
        path = os.path.join(pdf_dir, entry["pdf"])
        if not os.path.exists(path):
            return entry["pdf"], FileNotFoundError(path)
        try:
            return entry["pdf"], client.files.upload(file=path)
        except Exception as exc:
            return entry["pdf"], exc

    print(f"Uploading {len(GOLD)} PDFs in parallel …", flush=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = dict(ex.map(_upload, GOLD))
    ok = sum(1 for v in results.values() if not isinstance(v, Exception))
    print(f"  {ok}/{len(GOLD)} uploaded OK\n", flush=True)
    return results


def delete_all(client: genai.Client, file_map: dict) -> None:
    def _del(item):
        _, f = item
        if not isinstance(f, Exception):
            try:
                client.files.delete(name=f.name)
            except Exception:
                pass
    with ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(_del, file_map.items()))


def run_once(file_map: dict, max_workers: int) -> dict:
    """Grade all docs in parallel; return aggregate metrics for one trial."""
    def _grade(entry):
        f = file_map.get(entry["pdf"])
        if isinstance(f, Exception) or f is None:
            return entry, [], str(f or "not uploaded")
        results, err = pipe.grade_with_file(entry, genai.Client(), f)
        return entry, results, err

    total = match = 0
    corr_total = corr_match = 0
    signals_correct = 0
    actionable_correct = 0
    errored = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for entry, results, err in ex.map(_grade, GOLD):
            if err:
                errored += 1
                continue
            limited = entry["pdf"] in SCHEMA_LIMITED_PDFS
            signals_ok = True
            actionable_ok = True
            for r in results:
                total += 1
                if not limited:
                    corr_total += 1
                if r["status"] == pipe.MATCH:
                    match += 1
                    if not limited:
                        corr_match += 1
                if r["path"] == "announcement_signals" and r["status"] != pipe.MATCH:
                    signals_ok = False
                if r["path"] == "is_actionable_signal" and r["status"] != pipe.MATCH:
                    actionable_ok = False
            if signals_ok:
                signals_correct += 1
            if actionable_ok:
                actionable_correct += 1

    return {
        "overall_pct": round(100 * match / total, 1) if total else 0.0,
        "corrected_pct": round(100 * corr_match / corr_total, 1) if corr_total else 0.0,
        "signals_correct": signals_correct,
        "actionable_correct": actionable_correct,
        "n_docs": len(GOLD),
        "errored": errored,
    }


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    client = genai.Client()
    file_map = upload_all(client)

    try:
        summary = {}
        for condition in ["fewshot", "fewshot_rule"]:
            pipe.USE_FEWSHOT = True
            pipe.USE_ACTIONABLE_RULE = (condition == "fewshot_rule")
            runs = []
            for i in range(trials):
                r = run_once(file_map, max_workers)
                runs.append(r)
                print(f"[{condition}] run {i + 1}: "
                      f"actionable_ok {r['actionable_correct']}/{r['n_docs']}  "
                      f"signals_ok {r['signals_correct']}/{r['n_docs']}  "
                      f"overall {r['overall_pct']}%  (errored {r['errored']})", flush=True)
            summary[condition] = runs
    finally:
        print("\nCleaning up uploaded files …", flush=True)
        delete_all(client, file_map)

    print("\n=== COMPARISON (mean [min, max] over %d trials) ===" % trials)
    metrics = [
        ("actionable_correct", "Docs w/ is_actionable exactly right  [PRIMARY Exp2]"),
        ("signals_correct",    "Docs w/ signals exactly right"),
        ("corrected_pct",      "Corrected match % (excl. Ghani)"),
        ("overall_pct",        "Overall match %"),
    ]
    for key, label in metrics:
        print(f"\n{label}:")
        for condition in ["fewshot", "fewshot_rule"]:
            vals = [r[key] for r in summary[condition]]
            mean = round(statistics.mean(vals), 1)
            print(f"   {condition:9}: mean {mean}  range [{min(vals)}, {max(vals)}]")


if __name__ == "__main__":
    main()
