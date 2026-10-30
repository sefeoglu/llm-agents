"""LLM interface plus a deterministic rule-based stand-in (no API needed)."""
import re
from typing import List, Optional


class LLM:
    def decide(self, question: str, observations: List[dict]) -> dict:
        """Return {"tool": name, "arguments": {...}} or {"answer": str}."""
        raise NotImplementedError


class RuleBasedLLM(LLM):
    def decide(self, question: str, observations: List[dict]) -> dict:
        used = {o["tool"] for o in observations}
        expr = re.search(r"[\d(][\d\s+\-*/().^]*[\d)]", question)
        if expr and "calculator" not in used:
            return {"tool": "calculator", "arguments": {"expression": expr.group().replace("^", "**")}}
        if "search" not in used and not expr:
            return {"tool": "search", "arguments": {"query": question}}
        if not observations:
            return {"answer": "I don't know."}
        last = observations[-1]
        if last["tool"] == "calculator":
            return {"answer": str(last["result"])}
        docs = last["result"]
        return {"answer": docs[0][0] if docs else "No relevant documents found."}
