import json
import pipeline as pipe

pipe.USE_FEWSHOT = True
gold = json.load(open("gold_set.json"))
for entry in gold:
    results, err = pipe.grade_document(entry, ".")
    if err:
        print(f"{entry['pdf']:20}  ERROR: {err}")
        continue
    for r in results:
        if r["path"] == "announcement_signals" and r["status"] != pipe.MATCH:
            print(f"{entry['pdf']:20}  gold={r['gold']}  got={r['got']}")
