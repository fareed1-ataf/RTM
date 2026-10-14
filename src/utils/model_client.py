"""
src/utils/model_client.py
Unified LLM client — routes to Gemini or Ollama based on config.
Zero model names in code — everything comes from config/models.yaml.

Usage:
    from src.utils.config_loader import load_config
    from src.utils.model_client import ModelClient

    cfg = load_config()
    client = ModelClient(cfg.base_detector, cfg.secrets)
    response = client.generate(prompt="...", system_prompt="...")
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests

from src.utils.config_loader import ModelConfig, SecretsConfig
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class _BaseLLMClient(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Send a prompt and return the text response.

        Args:
            prompt: User message.
            system_prompt: Optional system instruction.

        Returns:
            Model response as a string.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return approximate token count for text."""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class _GeminiClient(_BaseLLMClient):
    """Google Gemini via google-generativeai SDK."""

    def __init__(self, model_cfg: ModelConfig, api_key: str) -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )
        if not api_key:
            raise ValueError(
                "[Config Error] gemini.api_key is empty in config/secrets.yaml.\n"
                "  -> Open config/secrets.yaml and set your API key."
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_cfg.model,
            generation_config=genai.types.GenerationConfig(
                temperature=model_cfg.temperature,
                max_output_tokens=model_cfg.max_output_tokens,
            ),
        )
        self._model_name = model_cfg.model
        log.debug(f"GeminiClient initialized: model={model_cfg.model}")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        t0 = time.time()
        response = self._model.generate_content(full_prompt)
        elapsed = (time.time() - t0) * 1000
        log.debug(f"Gemini call: {elapsed:.0f}ms, model={self._model_name}")
        return response.text.strip()

    def count_tokens(self, text: str) -> int:
        # Approximate: 1 token ~ 4 characters for English text
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

class _OllamaClient(_BaseLLMClient):
    """Local Ollama via HTTP REST API (localhost:11434)."""

    def __init__(self, model_cfg: ModelConfig, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model_cfg.model
        self._temperature = model_cfg.temperature
        self._max_tokens = model_cfg.max_output_tokens
        # Verify Ollama is reachable
        self._check_connection()
        log.debug(f"OllamaClient initialized: model={model_cfg.model}, url={base_url}")

    def _check_connection(self) -> None:
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"[Ollama Error] Cannot connect to Ollama at {self._base_url}.\n"
                f"  -> Make sure Ollama is running: run 'ollama serve' in a terminal."
            )
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(f"[Ollama Error] HTTP error: {e}")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        t0 = time.time()
        r = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        elapsed = (time.time() - t0) * 1000
        result = r.json().get("response", "").strip()
        log.debug(f"Ollama call: {elapsed:.0f}ms, model={self._model}")
        return result

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Public factory class
# ---------------------------------------------------------------------------

class ModelClient:
    """
    Public client — routes to Gemini or Ollama based on ModelConfig.provider.
    This is the ONLY class code outside utils/ should import.
    """

    def __init__(self, model_cfg: ModelConfig, secrets: SecretsConfig) -> None:
        provider = model_cfg.provider.lower()
        if provider == "gemini":
            self._client = _GeminiClient(model_cfg, secrets.gemini_api_key)
        elif provider == "ollama":
            self._client = _OllamaClient(model_cfg, secrets.ollama_base_url)
        else:
            raise ValueError(
                f"[Config Error] Unknown provider '{provider}' in models.yaml.\n"
                f"  -> Valid options: 'gemini' | 'ollama'"
            )
        self.provider = provider
        self.model_name = model_cfg.model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the configured model."""
        return self._client.generate(prompt, system_prompt)

    def count_tokens(self, text: str) -> int:
        """Return approximate token count."""
        return self._client.count_tokens(text)

    def __repr__(self) -> str:
        return f"ModelClient(provider={self.provider}, model={self.model_name})"
