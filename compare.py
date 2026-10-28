"""Compare two evaluated batches side by side (e.g. before and after a change).

Usage:
    python compare.py outputs/batch_<before> outputs/batch_<after>

Both batches must be evaluated with the same judge model and JUDGE_VERSION,
otherwise differences may come from the judge instead of the pipeline.
"""
import json
import sys
from pathlib import Path

from evaluate import summary_row


def load(batch: Path) -> dict[str, dict]:
    results = {}
    for f in sorted(batch.rglob("eval.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        results[r["qid"] or r["question"][:40]] = r
    return results


def totals(results: dict[str, dict]) -> dict:
    rows = [summary_row(r) for r in results.values()]
    rel = [r["_rel"] for r in rows]
    faith = [r["_faith"] for r in rows]

    def frac(pairs):
        den = sum(b for _, b in pairs)
        return sum(a for a, _ in pairs) / den if den else 0.0

    n = len(rows) or 1
    return {
        "relevant papers": frac(rel),
        "faithful summaries": frac(faith),
        "reports without unsupported claims": sum(r["_grounded"] for r in rows) / n,
        "reports not overstating fit": sum(r["_honest_fit"] for r in rows) / n,
        "prompt leaks (total)": sum(r["leaks"] for r in rows),
        "citation dumps (total)": sum(r["citation dumps"] for r in rows),
        "avg seconds per question": sum(r["seconds"] or 0 for r in rows) / n,
    }


def judges(results: dict[str, dict]) -> set:
    return {(r.get("judge_model"), r.get("judge_version")) for r in results.values()}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python compare.py <batch_before> <batch_after>")
    before, after = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    if not before or not after:
        raise SystemExit("Both folders need eval.json files. Run evaluate.py on them first.")
    if judges(before) != judges(after) or len(judges(before)) != 1:
        print(f"WARNING: different judges ({judges(before)} vs {judges(after)}), "
              f"the comparison is not reliable.\n")

    tb, ta = totals(before), totals(after)
    print(f"{'metric':<40}{'before':>10}{'after':>10}")
    for k in tb:
        if isinstance(tb[k], float) and k.startswith(("relevant", "faithful", "reports")):
            print(f"{k:<40}{tb[k]:>10.0%}{ta[k]:>10.0%}")
        elif isinstance(tb[k], float):
            print(f"{k:<40}{tb[k]:>10.1f}{ta[k]:>10.1f}")
        else:
            print(f"{k:<40}{tb[k]:>10}{ta[k]:>10}")

    print(f"\n{'question':<30}{'relevant':>16}{'unsupported':>14}{'overstates':>14}")
    for qid in sorted(set(before) | set(after)):
        b = summary_row(before[qid]) if qid in before else None
        a = summary_row(after[qid]) if qid in after else None

        def cell(row, key):
            return str(row[key]) if row else "-"
        print(f"{qid:<30}{cell(b, 'relevant') + ' -> ' + cell(a, 'relevant'):>16}"
              f"{cell(b, 'unsupported claims') + ' -> ' + cell(a, 'unsupported claims'):>14}"
              f"{cell(b, 'overstates fit') + ' -> ' + cell(a, 'overstates fit'):>14}")


if __name__ == "__main__":
    main()
