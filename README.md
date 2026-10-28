# arXiv Research Agent

An LLM workflow that turns a research question into a short, cited literature review
built from arXiv papers. It also comes with an evaluation suite to measure how good
those reviews are.

I'm building it alongside Andrew Ng's [Agentic AI course](https://www.deeplearning.ai/courses/agentic-ai)
(DeepLearning.AI). Each course module adds a new pattern to the project. The code uses plain Python and the
Gemini API, with no agent framework, so every step stays visible and testable.

```
python pipeline.py "How can retrieval-augmented generation systems be evaluated automatically?"
```

See [`examples/report_rag_evaluation.md`](examples/report_rag_evaluation.md) for an output.

## How it works (Module 1: task decomposition)

The task is split into five steps. Each one is done either by an LLM or by plain code:

| Step | Done by | What it does |
|---|---|---|
| 1. Search queries | LLM | Writes 3 queries as key phrases, each anchored on the core topic |
| 2. arXiv search | code | Turns phrases into arXiv query syntax (exact phrases, `AND`, title/abstract only, cs.CL / cs.AI / cs.LG) and drops duplicates |
| 3. Selection | LLM | Picks the 5 most relevant papers and gives a reason for each. Invalid picks are filtered out in code |
| 4. Summaries | LLM | Summarizes each abstract, using only what the abstract says |
| 5. Report | LLM + code | The LLM writes the review with inline citations. The reference list is built **by code** from arXiv metadata, so it can't be hallucinated |

Every run saves a `trace.json` with all intermediate outputs (queries, search log,
candidates, selection reasons, summaries, timings, LLM calls). This file is the basis
for error analysis.

**Reliability on a free API tier** (`llm.py`): a client-side rate limiter keeps requests
under the per-minute limit. Temporary errors are retried, using the server's `retryDelay`
for 429 responses. A fallback model plus a circuit breaker stop a failing model from
slowing down every later call.

## Evaluation

`questions.json` has 9 test questions. They cover narrow and broad topics, plus one
deliberately hard question with little literature (*Italian legal documents*).
`evaluate.py` scores each run in two ways:

- **Code-based checks (objective):** whether well-known papers were found, invalid or
  unused citations, uncited paragraphs, "citation dumps" (4+ sources cited at once),
  prompt text leaking into the output, and run time.
- **LLM-as-judge (yes/no questions, not 1–5 scores):** Is each selected paper relevant?
  Is each summary faithful to its abstract? Is each claim in the report supported by the
  sources it cites? Does the report present sources as more on-topic than they are?

```
python run_batch.py                       # run all test questions
python evaluate.py outputs/batch_<time>   # score them (resumes if the quota runs out)
python compare.py <batch_before> <batch_after>
```

## What error analysis found

**1. The first bug was in search, not in the LLM.** On the first run, the candidates for
a question about self-reflection in LLM agents included *"Quantum memories at finite
temperature"*. The trace showed why: plain keyword queries make arXiv match single words,
across all fields. Changing them to structured queries (exact phrases, `AND`, subject
filter) got the foundational paper (Reflexion) into the results.

**2. Selection works well. Search is the bottleneck.** In the baseline, every well-known
paper that made it into the candidate pool was selected (7/7). But only 7 of 17 were
found at all.

**3. A measured round of fixes.** Three prompt changes, each aimed at a failure seen in
the traces: anchor every query on the core topic, make summaries describe the paper
instead of commenting on "the abstract", and make the report describe partly-matching
sources honestly.

| Metric (9 questions) | Baseline | After fixes |
|---|---|---|
| Selected papers judged relevant | 87% | 93% |
| Summaries judged faithful | 93% | 100% |
| Prompt leaks (code check) | 11 | 1 |
| RAG-evaluation question: relevant papers | 2/5 | 5/5 (RAGAS and ARES now found) |
| Well-known papers found | 7/17 | 6/17 (no real change, see below) |

Full tables: [`examples/eval_baseline.md`](examples/eval_baseline.md),
[`examples/eval_after_fixes.md`](examples/eval_after_fixes.md).

Which well-known papers get found depends a lot on whether the model happens to write a
method name (e.g. "Toolformer") as a query in that run. With one run per question,
differences of ±1 paper are noise.

**4. The judge needs evaluating too.** I checked the judge's flags by hand against the
abstracts:
- The first judge version was **too strict**: of 5 "unsupported claims", 1 was a real
  error, 2 were minor citation issues and 2 were false alarms.
- After I made it reason first and then give an explicit verdict, it became **too
  lenient**. It marked a real misattribution as "supported", even though its own
  analysis said the review "slightly conflates" two results. It also missed a baseline
  report that presented a general legal-NLP paper as research on Italian law.

So the relevance and faithfulness numbers look trustworthy, but the "unsupported claims"
and "overstates fit" numbers are not yet. The judge is the smallest model and also wrote
the reports. Next step: calibrate it against a small set of hand-labelled cases.

## Roadmap (following the course)

- [x] **Module 1:** task decomposition, fixed pipeline, eval set, error analysis
- [ ] **Module 2:** reflection on the step where evals show the most errors
- [ ] **Module 3:** tool use (the model decides when to search or read full papers)
- [ ] **Module 4:** judge calibration against human labels, cost/latency tracking, model routing
- [ ] **Module 5:** planning and multi-agent roles (researcher, writer, editor)

## Setup

Requires Python 3.10+ and a free Gemini API key from [Google AI Studio](https://aistudio.google.com).

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env, then add your key
python smoke_test.py           # checks the LLM and arXiv connections
```

Use `python list_models.py` to see which models your key can use, and set the model
names and rate limit in `.env`.
