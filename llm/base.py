"""
Abstract Base Class for LLM Providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLMProvider(ABC):
    """Abstract interface for LLM backends."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of provider (e.g. 'groq', 'mock')."""
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Whether the provider operates in deterministic mock mode."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> str:
        """Generate a completion response given prompt and optional system context."""
        pass
