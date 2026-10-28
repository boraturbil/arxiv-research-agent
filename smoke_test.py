"""Quick check that the LLM connection and arXiv search both work."""
import arxiv

from llm import DEFAULT_MODEL, chat

print(f"Model: {DEFAULT_MODEL}")

# 1) LLM test
answer = chat("In one sentence, what is an agentic AI workflow?")
print("\nLLM answer:\n", answer)

# 2) arXiv test
search = arxiv.Search(query="agentic workflows reflection LLM", max_results=3,
                      sort_by=arxiv.SortCriterion.Relevance)
print("\narXiv results:")
for paper in arxiv.Client().results(search):
    print(f"- {paper.title} ({paper.published.year}) -> {paper.entry_id}")
