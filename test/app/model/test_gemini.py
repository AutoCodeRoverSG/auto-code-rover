import time
import os
import sys
import pytest
from unittest.mock import MagicMock
from tenacity import RetryError
from app.model.gemini import *
from app.model import common
from litellm.utils import ModelResponse, Choices, Message
from openai import BadRequestError
from app.log import log_and_print
from test.pytest_utils import *

# --- Dummy Utilities ---


def DummyGeminiResponse(content="Test response", input_tokens=1, output_tokens=2):
    """Return a dummy ModelResponse-like object."""
    dummy = MagicMock(spec=ModelResponse)
    dummy.choices = [Choices(message=Message(content=content))]
    Usage = type(
        "Usage", (), {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
    )
    dummy.usage = Usage()
    return dummy


# --- Gemini API Key and Setup Tests ---


def test_gemini_setup(monkeypatch):
    monkeypatch.setattr(GeminiPro, "check_api_key", dummy_check_api_key)
    model = GeminiPro()
    model.setup()
    assert model is not None


def test_gemini_check_api_key_success(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda key: "dummy-key")
    model = Gemini15Pro()
    key = model.check_api_key()
    assert key == "dummy-key"


def test_gemini_check_api_key_failure(monkeypatch, capsys):
    # Simulate missing GEMINI_API_KEY and GOOGLE_APPLICATION_CREDENTIALS.
    monkeypatch.setattr(os, "getenv", lambda key: "")
    monkeypatch.setattr(sys, "exit", dummy_sys_exit)
    model = Gemini15Pro()
    with pytest.raises(SysExitException):
        model.check_api_key()
    captured = capsys.readouterr().out
    assert (
        "Please set the GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS env var"
        in captured
    )


def test_extract_resp_content(monkeypatch):
    model = Gemini15Pro()
    # Test when content exists.
    msg = Message(content="Hello")
    assert model.extract_resp_content(msg) == "Hello"
    # Test when content is None.
    msg_none = Message(content=None)
    assert model.extract_resp_content(msg_none) == ""


# --- Define dictionary of Gemini models to test ---
gemini_models = {
    "GeminiPro": GeminiPro,
    "Gemini15Pro": Gemini15Pro,
}


@pytest.mark.parametrize(
    "model_class", gemini_models.values(), ids=gemini_models.keys()
)
def test_gemini_model_call(monkeypatch, model_class):
    """
    Test the normal call flow of Gemini models for both response formats.
    """
    # Patch sleep functions.
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    # Patch API key check and cost calculation.
    monkeypatch.setattr(GeminiModel, "check_api_key", dummy_check_api_key)
    monkeypatch.setattr(GeminiModel, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # ----- Case 1: response_format = "text" -----
    monkeypatch.setattr("litellm.completion", lambda **kwargs: DummyGeminiResponse())
    model = model_class()
    messages = [{"role": "user", "content": "Hello"}]
    # Call with response_format "text"
    content, cost, input_tokens, output_tokens = model.call(
        messages, response_format="text"
    )
    assert content == "Test response"
    assert cost == 0.5
    assert input_tokens == 1
    assert output_tokens == 2

    # ----- Case 2: response_format = "json_object" -----
    # Capture the messages passed to litellm.completion.
    captured_messages = []

    def dummy_completion(**kwargs):
        captured_messages.append(kwargs.get("messages"))
        return DummyGeminiResponse()

    monkeypatch.setattr("litellm.completion", dummy_completion)
    messages = [{"role": "user", "content": "Hello"}]
    # Call with response_format "json_object": GeminiModel.call will append a prefill message.
    content_json, cost_json, input_tokens_json, output_tokens_json = model.call(
        messages, response_format="json_object"
    )
    prefill = "{"
    # The extra message should be appended as the last message.
    modified_last = captured_messages[0][-1]["content"]
    assert prefill in modified_last
    # Also, if the returned content doesn't start with prefill, it is prepended.
    if not content_json.startswith(prefill):
        content_json = prefill + content_json
    assert content_json == "Test response" or content_json == prefill + "Test response"
    print(f"Gemini model {model_class.__name__} passed normal call flow tests.")
