"""
llm_generation/llm_interface.py
=================================
Ollama HTTP API client supporting Mistral, Qwen, and Llama 3.2.

Usage
-----
    from llm_generation.llm_interface import OllamaLLM, get_llm
    llm = get_llm("mistral", base_url="http://localhost:11434")
    answer = llm.generate("What is RAG?")
"""

from __future__ import annotations

import abc
import time
from typing import Optional

import requests
from loguru import logger


# ── Base Interface ─────────────────────────────────────────────────────────────

class BaseLLM(abc.ABC):
    """Abstract base class for all LLM backends."""

    name: str = "base"

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text response for *prompt*."""

    def generate_timed(self, prompt: str, **kwargs) -> dict:
        """Generate and also return latency + token estimates."""
        t0 = time.perf_counter()
        response = self.generate(prompt, **kwargs)
        elapsed_s = round(time.perf_counter() - t0, 4)
        return {
            "response": response,
            "latency_s": elapsed_s,
            "approx_tokens": len(response.split()),
        }


# ── Ollama LLM ─────────────────────────────────────────────────────────────────

class OllamaLLM(BaseLLM):
    """
    LLM client that calls the Ollama REST API.

    Parameters
    ----------
    model_name : Ollama model tag, e.g. 'mistral', 'qwen', 'llama3.2'
    base_url   : Ollama server URL (default http://localhost:11434)
    timeout    : request timeout in seconds
    max_tokens : maximum tokens to generate
    temperature: sampling temperature (0.0 = deterministic)
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """Call Ollama /api/generate and return the full response text."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Run: ollama serve"
            )
            return ""
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return ""
        except Exception as exc:
            logger.error(f"Ollama API error: {exc}")
            return ""

    def is_available(self) -> bool:
        """Check whether Ollama is reachable and the model is loaded."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            tags = resp.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in tags]
            if self.model_name not in model_names:
                logger.warning(
                    f"Model '{self.model_name}' not found in Ollama. "
                    f"Run: ollama pull {self.model_name}"
                )
                return False
            return True
        except Exception:
            return False


# ── Factory ────────────────────────────────────────────────────────────────────

_SUPPORTED_MODELS = {"mistral", "qwen", "llama3.2"}
_LLM_REGISTRY: dict[str, type[BaseLLM]] = {}


def get_llm(
    model_name: str,
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> BaseLLM:
    """
    Instantiate an LLM by model name.

    Parameters
    ----------
    model_name : 'mistral' | 'qwen' | 'llama3.2' (or any Ollama model tag)
    base_url   : Ollama server URL
    """
    if model_name in _LLM_REGISTRY:
        return _LLM_REGISTRY[model_name](
            model_name=model_name,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return OllamaLLM(
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def register_llm(name: str, cls: type[BaseLLM]) -> None:
    """Register a custom LLM class under *name*."""
    _LLM_REGISTRY[name] = cls
