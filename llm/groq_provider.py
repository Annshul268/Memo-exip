"""
Groq LLM Provider Integration.
Calls Groq Cloud API with OpenAI-compatible chat completion schema.
Gracefully handles missing keys, rate limits, and network errors.
"""

import json
import os
import requests
from typing import Optional
from llm.base import BaseLLMProvider


class GroqLLMProvider(BaseLLMProvider):
    """Integrates Groq API for high-speed inference."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1"
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.base_url = base_url.rstrip("/")
        
        if not self.api_key or self.api_key.startswith("your_"):
            raise ValueError("Invalid or missing GROQ_API_KEY. Configure .env with a valid key.")

    @property
    def provider_name(self) -> str:
        return f"groq:{self.model_name}"

    @property
    def is_mock(self) -> bool:
        return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            return f"[Groq Error: {str(e)}]"
