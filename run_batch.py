"""Run the pipeline on every question in questions.json.

Usage:
    python run_batch.py                 # all questions
    python run_batch.py q01 q04         # only questions whose id starts with these

Outputs go to outputs/batch_<timestamp>/<question id>/ (report.md + trace.json).
Then evaluate them with:
    python evaluate.py outputs/batch_<timestamp>
"""
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import llm
from pipeline import run


def main():
    data = json.loads(Path("questions.json").read_text(encoding="utf-8"))
    questions = data["questions"]
    filters = sys.argv[1:]
    if filters:
        questions = [q for q in questions if any(q["id"].startswith(f) for f in filters)]

    batch_dir = Path("outputs") / f"batch_{datetime.now():%Y%m%d-%H%M%S}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(questions)} questions with model {llm.DEFAULT_MODEL}")
    print(f"Saving to {batch_dir}\n")

    log = []
    t0 = time.perf_counter()
    for i, q in enumerate(questions, start=1):
        print(f"=== ({i}/{len(questions)}) {q['id']}: {q['question']}")
        try:
            run(q["question"], out_dir=batch_dir / q["id"], qid=q["id"])
            log.append({"id": q["id"], "status": "ok"})
        except Exception as e:  # one failing question shouldn't stop the batch
            traceback.print_exc()
            log.append({"id": q["id"], "status": "error", "error": repr(e)})
        print()

    (batch_dir / "batch_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    ok = sum(1 for r in log if r["status"] == "ok")
    print(f"Finished: {ok}/{len(log)} succeeded in {time.perf_counter() - t0:.0f}s")
    print(f"Next: python evaluate.py {batch_dir}")


if __name__ == "__main__":
    main()
