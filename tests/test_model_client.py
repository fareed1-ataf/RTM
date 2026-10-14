"""
tests/test_model_client.py
Unit tests for src/utils/model_client.py

Uses mocking — does NOT make real API calls.
Run: pytest tests/test_model_client.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ModelConfig, SecretsConfig
from src.utils.model_client import ModelClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def gemini_cfg():
    return ModelConfig(
        provider="gemini",
        model="gemini-2.0-flash",
        temperature=0,
        max_output_tokens=50,
        use_logprobs=False,
        score_mode="json_0_100",
    )


@pytest.fixture()
def ollama_cfg():
    return ModelConfig(
        provider="ollama",
        model="llama3.1:8b",
        temperature=0,
        max_output_tokens=50,
        score_mode="json_0_100",
    )


@pytest.fixture()
def secrets_with_key():
    return SecretsConfig(
        gemini_api_key="test_api_key_abc123",
        ollama_base_url="http://localhost:11434",
    )


@pytest.fixture()
def secrets_no_key():
    return SecretsConfig(
        gemini_api_key="",
        ollama_base_url="http://localhost:11434",
    )


# ---------------------------------------------------------------------------
# Tests: Gemini client
# ---------------------------------------------------------------------------

def test_gemini_client_raises_on_empty_key(gemini_cfg, secrets_no_key):
    """ModelClient must raise ValueError when Gemini API key is empty."""
    # Mock the import so we don't need real google-generativeai
    with patch.dict("sys.modules", {"google.generativeai": MagicMock()}):
        with pytest.raises(ValueError) as exc_info:
            ModelClient(gemini_cfg, secrets_no_key)
        assert "api_key" in str(exc_info.value).lower()


def test_gemini_client_generate_returns_string(gemini_cfg, secrets_with_key):
    """ModelClient.generate() must return a non-empty string (mocked)."""
    mock_genai = MagicMock()
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"verdict": "ALLOW", "score": 5}'
    mock_model_instance.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model_instance
    mock_genai.types.GenerationConfig.return_value = {}

    with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
        client = ModelClient(gemini_cfg, secrets_with_key)
        result = client.generate(prompt="Test prompt", system_prompt="System")

    assert isinstance(result, str)
    assert len(result) > 0


def test_gemini_client_temperature_zero_enforced(gemini_cfg, secrets_with_key):
    """Ensure temperature=0 is passed to the Gemini GenerationConfig."""
    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = MagicMock()
    mock_genai.types.GenerationConfig.return_value = {}

    with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
        ModelClient(gemini_cfg, secrets_with_key)

    # Check GenerationConfig was called with temperature=0
    call_kwargs = mock_genai.types.GenerationConfig.call_args[1]
    assert call_kwargs.get("temperature") == 0


# ---------------------------------------------------------------------------
# Tests: Ollama client
# ---------------------------------------------------------------------------

def test_ollama_client_generate_returns_string(ollama_cfg, secrets_with_key):
    """OllamaClient.generate() must return a string (mocked HTTP)."""
    import requests as req_module

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": '{"verdict": "BLOCK", "score": 95}'}
    mock_response.raise_for_status.return_value = None

    # Also mock the connection check
    mock_tags_response = MagicMock()
    mock_tags_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_tags_response), \
         patch("requests.post", return_value=mock_response):
        client = ModelClient(ollama_cfg, secrets_with_key)
        result = client.generate(prompt="Test prompt")

    assert isinstance(result, str)
    assert len(result) > 0


def test_ollama_client_raises_on_connection_error(ollama_cfg, secrets_with_key):
    """OllamaClient must raise ConnectionError when Ollama server is unreachable."""
    import requests

    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        with pytest.raises(ConnectionError) as exc_info:
            ModelClient(ollama_cfg, secrets_with_key)
        assert "ollama" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Tests: Unknown provider
# ---------------------------------------------------------------------------

def test_unknown_provider_raises_value_error(secrets_with_key):
    """ModelClient must raise ValueError for unknown provider."""
    bad_cfg = ModelConfig(
        provider="unknown_provider",
        model="some-model",
        temperature=0,
    )
    with pytest.raises(ValueError) as exc_info:
        ModelClient(bad_cfg, secrets_with_key)
    assert "provider" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Tests: ModelClient interface
# ---------------------------------------------------------------------------

def test_model_client_has_generate_method(gemini_cfg, secrets_with_key):
    """ModelClient must expose .generate() method."""
    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = MagicMock()
    mock_genai.types.GenerationConfig.return_value = {}

    with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
        client = ModelClient(gemini_cfg, secrets_with_key)

    assert callable(getattr(client, "generate", None))


def test_model_client_has_count_tokens_method(gemini_cfg, secrets_with_key):
    """ModelClient must expose .count_tokens() method."""
    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = MagicMock()
    mock_genai.types.GenerationConfig.return_value = {}

    with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
        client = ModelClient(gemini_cfg, secrets_with_key)

    assert callable(getattr(client, "count_tokens", None))
    result = client.count_tokens("Hello world this is a test")
    assert isinstance(result, int)
    assert result > 0


def test_model_client_repr(gemini_cfg, secrets_with_key):
    """ModelClient.__repr__ must include provider and model name."""
    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = MagicMock()
    mock_genai.types.GenerationConfig.return_value = {}

    with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
        client = ModelClient(gemini_cfg, secrets_with_key)

    r = repr(client)
    assert "gemini" in r.lower()
    assert "gemini-2.0-flash" in r
