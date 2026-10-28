"""Module 1: a fixed research pipeline (no agent autonomy yet).

    question -> [1] search queries -> [2] arXiv search -> [3] select papers
             -> [4] summarize each paper -> [5] write cited report

Usage:
    python pipeline.py "What are recent approaches to self-reflection in LLM agents?"

Each run saves two files in outputs/<timestamp>_<question>/:
    report.md   the final report
    trace.json  every intermediate output, for error analysis and evals
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import arxiv
from pydantic import BaseModel

import llm
from llm import chat, chat_json

QUERIES_COUNT = 3        # step 1: how many search queries to generate
RESULTS_PER_QUERY = 8    # step 2: arXiv results per query
PAPERS_TO_SELECT = 5     # step 3: how many papers go into the report

# Step 2 only searches these arXiv categories (computation & language, AI, machine learning).
# Without this filter, plain keyword search returned physics and unrelated LLM papers.
ARXIV_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str
    url: str


class Selection(BaseModel):
    index: int
    reason: str


# ---------- Step 1: generate search queries (LLM) ----------

def generate_queries(question: str, n: int = QUERIES_COUNT) -> list[list[str]]:
    """Return n queries, each a list of 1-3 key phrases that must ALL appear in a paper."""
    prompt = f"""You help search arXiv for papers that answer a research question.

Research question: {question}

Write {n} different search queries. Each query is a list of 1-3 short key phrases
(1-3 words each). A paper matches only if ALL phrases of a query appear in its
title or abstract, so keep each query focused and avoid generic words like
"large language model", "approach" or "recent".

Every query must contain one phrase that names the core topic of the question
(or a common synonym of it), so no query drifts to other fields. Add 1-2 phrases
for the specific angle. A query may also be just the name of a well-known method.

Use the established terminology researchers in this field actually use, including
names of well-known methods or benchmarks if you know them. Make the {n} queries
cover different angles or synonyms, so together they find a broad set of relevant papers.

Example for "How can chatbots be evaluated automatically?":
[["chatbot", "automatic evaluation"], ["dialogue systems", "evaluation metric"], ["USR metric"]]"""
    queries = chat_json(prompt, list[list[str]])
    cleaned = [[p.strip() for p in q if p.strip()][:3] for q in queries]
    return [q for q in cleaned if q][:n]


# ---------- Step 2: search arXiv (code) ----------

_arxiv_client = arxiv.Client(page_size=RESULTS_PER_QUERY, delay_seconds=3, num_retries=3)


def build_arxiv_query(phrases: list[str]) -> str:
    """Turn key phrases into arXiv query syntax.

    ["self-reflection", "language agents"] ->
    (ti:"self-reflection" OR abs:"self-reflection") AND (ti:"language agents" OR abs:"language agents")
    AND (cat:cs.CL OR cat:cs.AI OR cat:cs.LG)
    """
    parts = []
    for p in phrases:
        p = p.replace('"', "")
        parts.append(f'(ti:"{p}" OR abs:"{p}")')
    categories = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    return " AND ".join(parts) + f" AND ({categories})"


def search_arxiv(queries: list[list[str]], per_query: int = RESULTS_PER_QUERY,
                 log: list | None = None) -> list[Paper]:
    papers: dict[str, Paper] = {}  # keyed by id, so duplicates across queries are dropped
    for phrases in queries:
        # Try the strict query first; if it finds nothing, relax it by dropping
        # the last phrase (fewer phrases that must all match).
        results = []
        for n in range(len(phrases), 0, -1):
            arxiv_query = build_arxiv_query(phrases[:n])
            search = arxiv.Search(query=arxiv_query, max_results=per_query,
                                  sort_by=arxiv.SortCriterion.Relevance)
            results = list(_arxiv_client.results(search))
            if results:
                break
        if log is not None:  # recorded in trace.json for error analysis
            log.append({"phrases": phrases, "arxiv_query": arxiv_query,
                        "results": len(results)})
        for r in results:
            arxiv_id = re.sub(r"v\d+$", "", r.get_short_id())  # drop version suffix
            if arxiv_id not in papers:
                papers[arxiv_id] = Paper(
                    arxiv_id=arxiv_id,
                    title=" ".join(r.title.split()),
                    authors=[a.name for a in r.authors],
                    year=r.published.year,
                    abstract=" ".join(r.summary.split()),
                    url=r.entry_id,
                )
    return list(papers.values())


# ---------- Step 3: select the most relevant papers (LLM) ----------

def select_papers(question: str, candidates: list[Paper],
                  k: int = PAPERS_TO_SELECT) -> tuple[list[Paper], list[Selection]]:
    if len(candidates) <= k:
        picks = [Selection(index=i, reason="fewer candidates than k, all kept")
                 for i in range(len(candidates))]
        return candidates, picks

    listing = "\n\n".join(
        f"[{i}] {p.title} ({p.year})\n{p.abstract[:1000]}"
        for i, p in enumerate(candidates)
    )
    prompt = f"""Research question: {question}

Below are candidate papers from an arXiv search, each with an index number.
Choose the {k} papers that are most useful for answering the research question.
Prefer papers that directly address the question over loosely related ones.
For each choice give its index and a one-sentence reason.

Candidates:
{listing}"""
    picks = chat_json(prompt, list[Selection])

    # Don't trust the LLM blindly: drop invalid or repeated indices.
    valid, seen = [], set()
    for s in picks:
        if 0 <= s.index < len(candidates) and s.index not in seen:
            seen.add(s.index)
            valid.append(s)
    valid = valid[:k]
    if not valid:  # LLM returned nothing usable: fall back to search order
        valid = [Selection(index=i, reason="fallback: search order") for i in range(k)]
    return [candidates[s.index] for s in valid], valid


# ---------- Step 4: summarize each paper (LLM) ----------

def summarize_paper(question: str, paper: Paper) -> str:
    system = ("You summarize scientific papers accurately. Use only the information "
              "in the abstract you are given. Never add facts, numbers or claims. "
              "Write about the paper ('The paper proposes...'), never about the abstract "
              "or about these instructions.")
    prompt = f"""Research question: {question}

Paper: {paper.title} ({paper.year})
Abstract: {paper.abstract}

In 3-5 sentences, summarize what this paper does and what it finds, emphasizing
the parts related to the research question. Describe the paper for what it actually
studies, even if that differs from the question. Leave out anything the abstract
doesn't cover, without commenting on what is missing."""
    return chat(prompt, system=system).strip()


# ---------- Step 5: write the cited report (LLM + code) ----------

def write_report(question: str, papers: list[Paper], summaries: list[str]) -> str:
    sources = "\n\n".join(
        f"[{i}] {p.title} ({p.year})\nSummary: {s}"
        for i, (p, s) in enumerate(zip(papers, summaries), start=1)
    )
    prompt = f"""Write a short literature review in Markdown that answers this research question:

{question}

Rules:
- Use ONLY the sources below. Cite them inline by number, like [1] or [2][3].
- Every factual claim needs a citation. Don't invent sources, numbers or results.
- Cite the specific source(s) for each claim; avoid citing more than 3 sources at once.
- Some sources may only partly match the question (a general method, another domain or
  language). Describe each source for what it actually studies; never present it as being
  about the question's specific topic if it isn't. If the sources cover the question only
  partly, say so clearly in the Overview.
- Refer to the sources as "the reviewed papers", never as "the provided sources",
  "abstracts" or "summaries".
- Don't write a references section; it is added automatically.

Structure:
# <a title>
## Overview  (2-3 sentences that directly answer the question)
## Key findings  (group related papers and compare them)
## Open questions and gaps

Sources:
{sources}"""
    body = chat(prompt, temperature=0.4).strip()

    # The reference list is built by code, not by the LLM, so it can't be hallucinated.
    refs = "\n".join(
        f"[{i}] {', '.join(p.authors[:3])}{' et al.' if len(p.authors) > 3 else ''} "
        f"({p.year}). *{p.title}*. {p.url}"
        for i, p in enumerate(papers, start=1)
    )
    return f"{body}\n\n## References\n\n{refs}\n"


# ---------- Run the whole pipeline ----------

def _slug(text: str, max_len: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:max_len]


def run(question: str, out_dir: Path | None = None, qid: str | None = None) -> Path:
    """Run the pipeline once. Outputs go to out_dir (default: a new folder in outputs/)."""
    llm.CALL_LOG.clear()
    llm.UNAVAILABLE_MODELS.clear()  # give the main model a fresh chance on every run
    trace = {
        "qid": qid,
        "question": question,
        "model": llm.DEFAULT_MODEL,
        "fallback_model": llm.FALLBACK_MODEL,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "timings_s": {},
    }
    t0 = time.perf_counter()

    def timed(step, fn, *args):
        t = time.perf_counter()
        result = fn(*args)
        trace["timings_s"][step] = round(time.perf_counter() - t, 1)
        return result

    print("[1/5] Generating search queries...")
    queries = timed("1_queries", generate_queries, question)
    trace["queries"] = queries
    for q in queries:
        print(f"      - {' + '.join(q)}")

    print("[2/5] Searching arXiv...")
    trace["search_log"] = []
    candidates = timed("2_search", search_arxiv, queries, RESULTS_PER_QUERY,
                       trace["search_log"])
    for entry in trace["search_log"]:
        print(f"      {entry['results']:>2} results for: {' + '.join(entry['phrases'])}")
    trace["candidates"] = [{"arxiv_id": p.arxiv_id, "title": p.title, "year": p.year}
                           for p in candidates]
    print(f"      {len(candidates)} unique papers found")
    if not candidates:
        raise SystemExit("No papers found. Try rephrasing the question.")

    print("[3/5] Selecting the most relevant papers...")
    selected, picks = timed("3_select", select_papers, question, candidates)
    trace["selection"] = [{"arxiv_id": p.arxiv_id, "title": p.title, "reason": s.reason}
                          for p, s in zip(selected, picks)]
    for p in selected:
        print(f"      - {p.title[:80]} ({p.year})")

    print("[4/5] Summarizing papers...")
    t = time.perf_counter()
    summaries = []
    for i, p in enumerate(selected, start=1):
        print(f"      ({i}/{len(selected)}) {p.title[:70]}")
        summaries.append(summarize_paper(question, p))
    trace["timings_s"]["4_summarize"] = round(time.perf_counter() - t, 1)
    # Abstracts are saved next to summaries so evals can check faithfulness later.
    trace["summaries"] = [{"arxiv_id": p.arxiv_id, "abstract": p.abstract, "summary": s}
                          for p, s in zip(selected, summaries)]

    print("[5/5] Writing the report...")
    report = timed("5_report", write_report, question, selected, summaries)
    trace["report"] = report
    trace["timings_s"]["total"] = round(time.perf_counter() - t0, 1)
    trace["llm_calls"] = llm.CALL_LOG.copy()

    if out_dir is None:
        out_dir = Path("outputs") / f"{datetime.now():%Y%m%d-%H%M%S}_{_slug(question)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    print(f"\nDone in {trace['timings_s']['total']}s, {len(llm.CALL_LOG)} LLM calls.")
    print(f"Saved to {out_dir}")
    return out_dir


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Research question: ")
    run(question)
