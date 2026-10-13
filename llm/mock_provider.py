"""
Deterministic Mock LLM Provider.
Enables local development, automated testing, and reproducible evaluations
without external API keys or network dependencies.
"""

import re
from typing import Optional
from llm.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic provider that parses memories injected into the prompt
    and synthesizes an accurate, grounded agent response.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def is_mock(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> str:
        """
        Parses injected memory context to answer user queries deterministically.
        Extracts subject, attribute, and value from context blocks.
        """
        # Extract memory context block if present
        context_match = re.search(r"### Retrieved Memories:\n(.*?)(?=\n###|\Z)", prompt, re.DOTALL)
        memory_lines = []
        if context_match:
            raw_context = context_match.group(1).strip()
            memory_lines = [line.strip("- * \t") for line in raw_context.split("\n") if line.strip()]

        # Extract user query
        query_match = re.search(r"### User Query:\n(.*?)(?=\n###|\Z)", prompt, re.DOTALL)
        query = query_match.group(1).strip() if query_match else prompt

        if not memory_lines:
            return "[Mock LLM] I do not have any relevant memories or information stored to answer this query."

        # Parse memory lines of the format: "subject -> attribute: value [Valid from ...] (Status: ...)"
        parsed_facts = []
        for line in memory_lines:
            match = re.match(r"(.*?)\s*->\s*(.*?):\s*([^\s\[]+(?:\s+[^\s\[]+)*)", line)
            if match:
                s, a, v = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                status_m = re.search(r"\(Status:\s*(\w+)\)", line)
                status = status_m.group(1) if status_m else "current"
                parsed_facts.append({"subject": s, "attribute": a, "value": v, "status": status, "raw": line})

        # Answer formulation based on parsed facts
        q_lower = query.lower()
        
        # Check timeline queries
        if any(w in q_lower for w in ["history", "timeline", "progression", "change"]):
            if parsed_facts:
                resp = "[Mock LLM] Based on stored temporal records, here is the history:\n"
                for idx, f in enumerate(parsed_facts, 1):
                    resp += f"{idx}. {f['subject']} {f['attribute']} was '{f['value']}' ({f['status']}).\n"
                return resp.strip()

        # Check historical query intent
        is_historical_q = any(w in q_lower for w in ["before", "previously", "earlier", "past", "was", "old", "former"])
        
        if is_historical_q:
            # Look for superseded facts first
            superseded_facts = [f for f in parsed_facts if f["status"] == "superseded"]
            if superseded_facts:
                chosen = superseded_facts[0]
                return f"[Mock LLM] Previously, the {chosen['attribute']} was '{chosen['value']}'."
            elif parsed_facts:
                chosen = parsed_facts[0]
                return f"[Mock LLM] The recorded {chosen['attribute']} was '{chosen['value']}'."

        # Check current query intent
        current_facts = [f for f in parsed_facts if f["status"] == "current"]
        if current_facts:
            chosen = current_facts[0]
            return f"[Mock LLM] Currently, the {chosen['attribute']} is '{chosen['value']}'."

        # Default fallback to top fact
        if parsed_facts:
            chosen = parsed_facts[0]
            return f"[Mock LLM] According to my memory, the {chosen['attribute']} is '{chosen['value']}'."

        return f"[Mock LLM] Based on retrieved context: {memory_lines[0]}"
