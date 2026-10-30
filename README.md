# llm-agents
LLM Agents using orchestration and tool calling.

- `llm_agents/tools.py`: tool registry and tool calling (calculator, search)
- `llm_agents/reranker.py`: BM25 reranker of retrieved documents
- `llm_agents/orchestrator.py`: agent loop orchestrating LLM, tools and reranker
- `llm_agents/llm.py`: `LLM` interface and an offline `RuleBasedLLM`; plug in a real LLM by subclassing `LLM`

Run `python main.py` for a demo and `python -m pytest` for tests.
