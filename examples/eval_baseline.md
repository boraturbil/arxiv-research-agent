# Evaluation summary: baseline (before fixes)

| id | candidates | canonical (cand / sel) | relevant | faithful summaries | unsupported claims | minor citation issues | overstates fit | uncited paragraphs | citation dumps | leaks | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q01_self_reflection | 23 | 1/3 / 1/3 | 5/5 | 5/5 | 0 | 0 | no | 0 | 1 | 0 | 36.0 |
| q02_tool_use | 22 | 1/3 / 1/3 | 5/5 | 5/5 | 0 | 0 | no | 0 | 1 | 2 | 42.6 |
| q03_rag_eval | 18 | 0/2 / 0/2 | 2/5 | 5/5 | 0 | 0 | no | 0 | 0 | 1 | 29.6 |
| q04_llm_judge | 24 | 0/2 / 0/2 | 5/5 | 5/5 | 0 | 0 | no | 0 | 1 | 0 | 42.9 |
| q05_multiagent_debate | 15 | 1/1 / 1/1 | 5/5 | 4/5 | 0 | 0 | no | 0 | 0 | 2 | 29.3 |
| q06_reasoning_prompting | 23 | 2/3 / 2/3 | 5/5 | 5/5 | 0 | 0 | no | 0 | 1 | 0 | 42.1 |
| q07_hallucination_detection | 14 | 1/1 / 1/1 | 5/5 | 5/5 | 0 | 0 | no | 0 | 2 | 0 | 35.8 |
| q08_model_routing | 22 | 1/2 / 1/2 | 5/5 | 5/5 | 0 | 0 | no | 0 | 1 | 2 | 36.3 |
| q09_italian_legal | 18 | - | 2/5 | 3/5 | 0 | 0 | no | 0 | 0 | 4 | 32.5 |

## Totals

- Selected papers judged relevant: 87% (39/45)
- Summaries judged faithful: 93% (42/45)
- Reports with no unsupported claims: 100% (9/9)
- Reports that don't overstate how well sources fit the question: 100% (9/9)

Judge: gemini-3.5-flash-lite, judge version v2. See the README for why the two grounding/fit rows should not be trusted yet.
