"""Evaluate pipeline runs: code-based checks + LLM-as-judge.

Usage:
    python evaluate.py outputs/batch_<timestamp>      # every run in a batch
    python evaluate.py outputs/<one run folder>       # a single run
    python evaluate.py outputs/batch_<timestamp> --force   # re-judge runs already evaluated

Runs that already have an eval.json from the same judge model and JUDGE_VERSION are
skipped (older judge versions are re-judged automatically), so if the
quota runs out halfway, just run the same command again later to finish.

For each run it writes eval.json next to trace.json, and for the whole folder
eval_summary.md with one row per question.

Following Module 1: objective criteria are checked with code, subjective ones with an
LLM judge that answers yes/no questions (more reliable than 1-5 scores).
Judge model: JUDGE_MODEL in .env, or the main model if not set. The judge never switches
to the fallback model, so all scores come from the same judge. Spot-check the judge's
reasons against your own reading before trusting its numbers.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

from google.genai import errors
from pydantic import BaseModel

import llm
from llm import chat_json

JUDGE_MODEL = os.getenv("JUDGE_MODEL") or llm.DEFAULT_MODEL
# Bump this whenever a judge prompt or schema changes: scores from different judge
# versions are not comparable, so runs judged with an older version get re-judged.
JUDGE_VERSION = "v2"

# Phrases that show prompt instructions leaking into the output text.
# Summaries should describe the paper, so ANY mention of "the abstract" is a leak there.
SUMMARY_LEAK_PATTERNS = [
    r"based on the provided",
    r"\bthe (provided )?abstract\b",
    r"\bthis abstract\b",
]
# Reports may honestly say what the sources do or don't cover ("the reviewed papers only
# partly address..."). Only phrases that echo our prompt count as leaks there.
REPORT_LEAK_PATTERNS = [
    r"based on the provided",
    r"\bthe provided (abstracts?|summar\w*)",
    r"according to the (provided )?(abstracts?|summar\w*)",
    r"\bthe (abstracts?|summar\w*) (do|does|did) not\b",
]


# ---------- Code-based checks (objective) ----------

def report_body(report: str) -> str:
    return report.split("## References")[0]


def citation_checks(report: str, n_sources: int) -> dict:
    body = report_body(report)
    cited = [int(c) for c in re.findall(r"\[(\d+)\]", body)]
    invalid = sorted({c for c in cited if not 1 <= c <= n_sources})
    uncited_sources = sorted(set(range(1, n_sources + 1)) - set(cited))

    # Paragraphs of real content (not headings / notes) without any citation
    paragraphs = [p.strip() for p in body.split("\n\n")]
    content = [p for p in paragraphs if p and not p.startswith("#") and len(p) > 80]
    uncited_paragraphs = [p[:120] for p in content if not re.search(r"\[\d+\]", p)]

    # "Citation dumps": one sentence citing 4+ sources at once (hard to check what came from where)
    dumps = re.findall(r"(?:\[\d+\]\s*){4,}", body)

    return {
        "invalid_citations": invalid,
        "sources_never_cited": uncited_sources,
        "uncited_paragraphs": uncited_paragraphs,
        "citation_dumps": len(dumps),
    }


def leak_checks(trace: dict) -> list[str]:
    """Return one snippet per leaking text (each summary / the report counted at most once)."""
    texts = [(s["summary"], SUMMARY_LEAK_PATTERNS) for s in trace["summaries"]]
    texts.append((report_body(trace["report"]), REPORT_LEAK_PATTERNS))
    found = []
    for text, patterns in texts:
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                found.append(text[max(0, m.start() - 40):m.end() + 40].replace("\n", " "))
                break
    return found


def canonical_recall(trace: dict, expected: dict) -> dict:
    if not expected:
        return {}
    cand_ids = {c["arxiv_id"] for c in trace["candidates"]}
    sel_ids = {s["arxiv_id"] for s in trace["selection"]}
    return {
        "expected": expected,
        "found_in_candidates": [expected[i] for i in expected if i in cand_ids],
        "found_in_selection": [expected[i] for i in expected if i in sel_ids],
    }


# ---------- LLM-as-judge checks (subjective, yes/no) ----------

class RelevanceVerdict(BaseModel):
    index: int
    relevant: bool
    reason: str


class FaithfulnessVerdict(BaseModel):
    unsupported_claims: list[str]
    faithful: bool


class ClaimCheck(BaseModel):
    claim: str
    cited_sources: str
    analysis: str  # reasoning comes BEFORE the verdict, so the verdict follows from it
    verdict: Literal["supported", "minor_issue", "unsupported"]


class GroundingVerdict(BaseModel):
    checked_claims: list[ClaimCheck]


class FitVerdict(BaseModel):
    analysis: str
    misrepresented_sources: list[int]
    overstates_fit: bool


JUDGE_SYSTEM = ("You are a strict, careful reviewer of scientific literature reviews. "
                "Judge only from the text you are given. When unsure, say so in the reason.")


def judge_relevance(question: str, papers: list[dict]) -> list[RelevanceVerdict]:
    listing = "\n\n".join(f"[{i}] {p['title']}\n{p['abstract'][:1500]}"
                          for i, p in enumerate(papers))
    prompt = f"""Research question: {question}

For each paper below, decide whether it is RELEVANT: its main contribution directly
addresses the research question. A paper that only shares keywords, applies the topic in
an unrelated domain as a side detail, or is about a neighbouring topic is NOT relevant.
Give the index, a yes/no verdict and a one-sentence reason.

Papers:
{listing}"""
    return chat_json(prompt, list[RelevanceVerdict], system=JUDGE_SYSTEM, model=JUDGE_MODEL,
                     allow_fallback=False)


def judge_faithfulness(abstract: str, summary: str) -> FaithfulnessVerdict:
    prompt = f"""Check whether a summary is faithful to the abstract it was written from.

List every claim in the SUMMARY that is NOT supported by the ABSTRACT: invented facts,
wrong numbers, exaggerations, or claims the abstract doesn't make. Leaving things out is
fine; only list things that are added or changed. Sentences about the abstract itself
(e.g. "the abstract does not mention X") are not claims about the paper: ignore them,
they are counted by a separate check. faithful = true only if the list is empty.

ABSTRACT:
{abstract}

SUMMARY:
{summary}"""
    return chat_json(prompt, FaithfulnessVerdict, system=JUDGE_SYSTEM, model=JUDGE_MODEL,
                     allow_fallback=False)


def judge_grounding(report: str, papers: list[dict]) -> GroundingVerdict:
    sources = "\n\n".join(f"[{i}] {p['title']}\n{p['abstract']}"
                          for i, p in enumerate(papers, start=1))
    prompt = f"""Below is a literature review and the abstracts of the sources it cites by number.

Check the specific factual claims in the review (methods, findings, numbers, comparisons)
against the abstract(s) they cite. List every claim that could be a problem. For each one,
FIRST write your analysis comparing it with the cited abstract(s), THEN give a verdict:
- "supported": the cited abstracts back it up (if your analysis ends here, use this verdict)
- "minor_issue": essentially correct, but the citation is imprecise (e.g. one of several
  cited sources doesn't contain it, or the claim is in an uncited source)
- "unsupported": wrong, invented, attributed to the wrong cause or method, overstated,
  or a factual claim without any citation
Statements that only summarize or connect cited claims are fine and need not be listed.

REVIEW:
{report_body(report)}

SOURCES:
{sources}"""
    return chat_json(prompt, GroundingVerdict, system=JUDGE_SYSTEM, model=JUDGE_MODEL,
                     allow_fallback=False)


def judge_fit(question: str, report: str, papers: list[dict]) -> FitVerdict:
    sources = "\n\n".join(f"[{i}] {p['title']}\n{p['abstract']}"
                          for i, p in enumerate(papers, start=1))
    prompt = f"""Research question: {question}

Below is a literature review written to answer this question, and the abstracts of its sources.
Some sources may only partly address the question: a general method, a different domain,
language or task. A good review describes each source accurately for what it actually
studies, and says so when the sources only partly cover the question.

Check whether the review presents any source as being about the question's specific topic
when its abstract shows it is not, or suggests the topic is well covered when most sources
are off-topic. List the numbers of misrepresented sources.
overstates_fit = true if at least one source is misrepresented this way.

REVIEW:
{report_body(report)}

SOURCES:
{sources}"""
    return chat_json(prompt, FitVerdict, system=JUDGE_SYSTEM, model=JUDGE_MODEL,
                     allow_fallback=False)


# ---------- Run all checks on one run folder ----------

def run_code_checks(trace: dict, expected: dict) -> dict:
    """Objective checks: no API calls, so they are recomputed on every evaluate run."""
    return {
        "candidates": len(trace["candidates"]),
        "empty_queries": [e["phrases"] for e in trace.get("search_log", [])
                          if e["results"] == 0],
        "canonical_recall": canonical_recall(trace, expected),
        "citations": citation_checks(trace["report"], len(trace["summaries"])),
        "prompt_leaks": leak_checks(trace),
        "seconds": trace["timings_s"].get("total"),
        "retries": sum(c["retries"] for c in trace.get("llm_calls", [])),
    }


def expected_for(trace: dict, expected_by_id: dict, expected_by_question: dict) -> dict:
    return (expected_by_id.get(trace.get("qid")) or
            expected_by_question.get(trace["question"]) or {})

def evaluate_run(run_dir: Path, expected_by_id: dict, expected_by_question: dict) -> dict:
    trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    question = trace["question"]
    expected = expected_for(trace, expected_by_id, expected_by_question)

    # Abstracts of the selected papers are stored with the summaries
    papers = [{"title": sel["title"], "abstract": summ["abstract"]}
              for sel, summ in zip(trace["selection"], trace["summaries"])]

    n_calls_before = len(llm.CALL_LOG)
    print(f"  judging relevance...")
    relevance = judge_relevance(question, papers)
    print(f"  judging {len(papers)} summaries...")
    faithfulness = [judge_faithfulness(s["abstract"], s["summary"]) for s in trace["summaries"]]
    print(f"  judging report grounding...")
    grounding = judge_grounding(trace["report"], papers)
    print(f"  judging whether the report fits its sources to the question...")
    fit = judge_fit(question, trace["report"], papers)

    result = {
        "qid": trace.get("qid"),
        "question": question,
        "model": trace.get("model"),
        "judge_model": JUDGE_MODEL,
        "judge_version": JUDGE_VERSION,
        # Models that actually answered (should always be just the judge model)
        "judge_models_used": sorted({c["model"] for c in llm.CALL_LOG[n_calls_before:]}),
        "code_checks": run_code_checks(trace, expected),
        "judge": {
            "relevance": [{"title": papers[v.index]["title"] if 0 <= v.index < len(papers) else "?",
                           **v.model_dump()} for v in relevance],
            "faithfulness": [{"title": p["title"], **f.model_dump()}
                             for p, f in zip(papers, faithfulness)],
            "grounding": grounding.model_dump(),
            "fit": fit.model_dump(),
        },
    }
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    return result


# ---------- Summary table ----------

def summary_row(r: dict) -> dict:
    cc, j = r["code_checks"], r["judge"]
    rec = cc["canonical_recall"]
    n_rel = sum(v["relevant"] for v in j["relevance"])
    n_faith = sum(f["faithful"] for f in j["faithfulness"])
    verdicts = [c["verdict"] for c in j["grounding"]["checked_claims"]]
    n_unsup = verdicts.count("unsupported")
    n_minor = verdicts.count("minor_issue")
    overstates = j["fit"]["overstates_fit"]
    return {
        "id": r["qid"] or r["question"][:40],
        "candidates": cc["candidates"],
        "canonical (cand / sel)": (f"{len(rec['found_in_candidates'])}/{len(rec['expected'])} / "
                                   f"{len(rec['found_in_selection'])}/{len(rec['expected'])}"
                                   if rec else "-"),
        "relevant": f"{n_rel}/{len(j['relevance'])}",
        "faithful summaries": f"{n_faith}/{len(j['faithfulness'])}",
        "unsupported claims": n_unsup,
        "minor citation issues": n_minor,
        "overstates fit": "yes" if overstates else "no",
        "uncited paragraphs": len(cc["citations"]["uncited_paragraphs"]),
        "citation dumps": cc["citations"]["citation_dumps"],
        "leaks": len(cc["prompt_leaks"]),
        "seconds": cc["seconds"],
        "_rel": (n_rel, len(j["relevance"])),
        "_faith": (n_faith, len(j["faithfulness"])),
        "_grounded": n_unsup == 0,
        "_honest_fit": not overstates,
    }


def write_summary(results: list[dict], out_path: Path) -> str:
    rows = [summary_row(r) for r in results]
    cols = [k for k in rows[0] if not k.startswith("_")]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    def pct(pairs):
        num, den = sum(a for a, _ in pairs), sum(b for _, b in pairs)
        return f"{100 * num / den:.0f}% ({num}/{den})" if den else "-"

    totals = [
        f"- Selected papers judged relevant: {pct([r['_rel'] for r in rows])}",
        f"- Summaries judged faithful: {pct([r['_faith'] for r in rows])}",
        f"- Reports with no unsupported claims: "
        f"{pct([(int(r['_grounded']), 1) for r in rows])}",
        f"- Reports that don't overstate how well sources fit the question: "
        f"{pct([(int(r['_honest_fit']), 1) for r in rows])}",
    ]
    text = ("# Evaluation summary\n\n" + "\n".join(lines) + "\n\n## Totals\n\n" +
            "\n".join(totals) + "\n\nDetails per question: eval.json in each run folder.\n")
    out_path.write_text(text, encoding="utf-8")
    return text


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python evaluate.py <batch or run folder>")
    root = Path(sys.argv[1])
    run_dirs = sorted(p.parent for p in root.rglob("trace.json"))
    if not run_dirs:
        raise SystemExit(f"No trace.json found under {root}")

    expected_by_id, expected_by_question = {}, {}
    if Path("questions.json").exists():
        for q in json.loads(Path("questions.json").read_text(encoding="utf-8"))["questions"]:
            expected_by_id[q["id"]] = q.get("expected_papers", {})
            expected_by_question[q["question"]] = q.get("expected_papers", {})

    force = "--force" in sys.argv
    print(f"Evaluating {len(run_dirs)} run(s) with judge model {JUDGE_MODEL}\n")
    results = []
    for d in run_dirs:
        print(f"- {d.name}")
        existing = d / "eval.json"
        if existing.exists() and not force:
            old = json.loads(existing.read_text(encoding="utf-8"))
            used = old.get("judge_models_used", [old.get("judge_model")])
            if used == [JUDGE_MODEL] and old.get("judge_version") == JUDGE_VERSION:
                # same judge model and prompts: reuse, don't pay twice
                trace = json.loads((d / "trace.json").read_text(encoding="utf-8"))
                old["code_checks"] = run_code_checks(
                    trace, expected_for(trace, expected_by_id, expected_by_question))
                existing.write_text(json.dumps(old, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
                print("  judge results reused, code checks refreshed (--force to re-judge)")
                results.append(old)
                continue
        try:
            results.append(evaluate_run(d, expected_by_id, expected_by_question))
        except errors.APIError as e:
            if e.code == 429:
                print(f"  judge model {JUDGE_MODEL} is out of quota.\n"
                      f"  Stopping here. Run the same command later: finished runs are skipped.")
                break
            print(f"  FAILED: {e!r}")
        except Exception as e:
            print(f"  FAILED: {e!r}")

    if results:
        print("\n" + write_summary(results, root / "eval_summary.md"))
        print(f"Saved {root / 'eval_summary.md'}")


if __name__ == "__main__":
    main()
