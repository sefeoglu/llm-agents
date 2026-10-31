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
        has_math_intent = bool(re.search(r"\b(compute|calculate|evaluate)\b", question, re.I))
        has_operator = bool(re.search(r"\d\s*[+*/%^]\s*\d|\d\s*-\s*\d", question))
        is_calculation = expr and (has_math_intent or has_operator)
        if is_calculation and "calculator" not in used:
            return {"tool": "calculator", "arguments": {"expression": expr.group().replace("^", "**")}}
        if "search" not in used and not is_calculation:
            return {"tool": "search", "arguments": {"query": question}}
        if not observations:
            return {"answer": "I don't know."}
        last = observations[-1]
        if last["tool"] == "calculator":
            return {"answer": str(last["result"])}
        docs = last["result"]
        return {"answer": docs[0][0] if docs else "No relevant documents found."}
