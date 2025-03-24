import time
import sys
import os
import pytest
from unittest.mock import MagicMock
from tenacity import RetryError
from app.model.claude import *
from app.model import common
from litellm.utils import ModelResponse, Choices, Message
from litellm.exceptions import ContentPolicyViolationError
from openai import BadRequestError
from app.model.common import ClaudeContentPolicyViolation
from test.pytest_utils import *


# --- Dummy Response for litellm.completion ---
def DummyLiteLLMResponse(content="Test response", input_tokens=1, output_tokens=2):
    # Create a dummy object that satisfies ModelResponse using MagicMock.
    dummy = MagicMock(spec=ModelResponse)
    dummy.choices = [Choices(message=Message(content=content))]
    # Create a dummy usage object with required token counts.
    Usage = type(
        "Usage", (), {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
    )
    dummy.usage = Usage()
    return dummy


# --- Common model tests (i.e. checking API key for diff models) -> extract these to pytest_utils.py(?) ---
def test_setup(monkeypatch):
    # Patch check_api_key to return a dummy key.
    monkeypatch.setattr(Claude3_5Sonnet, "check_api_key", dummy_check_api_key)
    model = Claude3_5Sonnet()
    model.setup()

    assert model is not None


def test_check_api_key_success(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda key: "dummy-key")
    model = Claude3_5Sonnet()
    key = model.check_api_key()
    assert key == "dummy-key"


def test_check_api_key_failure(monkeypatch, capsys):
    # Simulate missing ANTHROPIC_API_KEY.
    monkeypatch.setattr(os, "getenv", lambda key: "")
    monkeypatch.setattr(sys, "exit", dummy_sys_exit)
    model = Claude3_5Sonnet()
    with pytest.raises(SysExitException):
        model.check_api_key()
    captured = capsys.readouterr().out
    assert "Please set the ANTHROPIC_API_KEY env var" in captured


def test_extract_resp_content(monkeypatch):
    model = Claude3_5Sonnet()
    # Test when content exists.
    msg = DummyMessage(content="Hello")
    assert model.extract_resp_content(msg) == "Hello"
    # Test when content is None.
    msg_none = DummyMessage(content=None)
    assert model.extract_resp_content(msg_none) == ""


# --- Define dictionary of Claude models to test ---
claude_models = {
    "Claude3Opus": Claude3Opus,
    "Claude3Sonnet": Claude3Sonnet,
    "Claude3Haiku": Claude3Haiku,
    "Claude3_5Sonnet": Claude3_5Sonnet,
    "Claude3_5SonnetNew": Claude3_5SonnetNew,
}


@pytest.mark.parametrize(
    "model_class", claude_models.values(), ids=claude_models.keys()
)
def test_anthropic_model_call(monkeypatch, model_class):
    """
    Test the normal call flow of Anthropic models for both response formats.
    """
    # Patch sleep functions.
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    # Patch check_api_key and calc_cost.
    monkeypatch.setattr(AnthropicModel, "check_api_key", dummy_check_api_key)
    monkeypatch.setattr(AnthropicModel, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # ----- Case 1: response_format = "text" -----
    monkeypatch.setattr("litellm.completion", lambda **kwargs: DummyLiteLLMResponse())
    model = model_class()
    messages = [{"role": "user", "content": "Hello"}]
    content, cost, input_tokens, output_tokens = model.call(
        messages, response_format="text"
    )
    assert content == "Test response"
    assert cost == 0.5
    assert input_tokens == 1
    assert output_tokens == 2

    # ----- Case 2: response_format = "json_object" -----
    # We'll capture the messages passed to litellm.completion.
    captured_messages = []

    def dummy_completion(**kwargs):
        captured_messages.append(kwargs.get("messages"))
        return DummyLiteLLMResponse()

    monkeypatch.setattr("litellm.completion", dummy_completion)
    messages = [{"role": "user", "content": "Hello"}]
    # Call with response_format "json_object" so the branch is triggered.
    content_json, cost_json, input_tokens_json, output_tokens_json = model.call(
        messages, response_format="json_object"
    )
    # The source code appends the following extra text to the last message:
    extra_text = "\nYour response should start with { and end with }. DO NOT write anything else other than the json."
    # Capture the modified message that was passed to litellm.completion.
    modified_last = captured_messages[0][-1]["content"]
    assert extra_text in modified_last
    # The dummy response remains the same.
    assert content_json == "Test response"
    print(f"Test passed for {model_class.__name__} with both response formats.")


# --- Test Content Policy Violation Handling ---
def test_claude_content_policy_violation(monkeypatch):
    """
    Test that if litellm.completion raises ContentPolicyViolationError,
    the call method logs and then raises ClaudeContentPolicyViolation.
    """
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    monkeypatch.setattr(AnthropicModel, "check_api_key", dummy_check_api_key)
    monkeypatch.setattr(AnthropicModel, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # Construct a ContentPolicyViolationError with dummy required args.
    def raise_cpv(*args, **kwargs):
        raise ContentPolicyViolationError("dummy_violation", "arg2", "arg3")

    monkeypatch.setattr("litellm.completion", raise_cpv)

    model = Claude3Opus()
    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(ClaudeContentPolicyViolation):
        model.call(messages, temperature=1.0)
    print("Claude content policy violation test passed.")


# --- Test BadRequestError Handling ---
@pytest.mark.parametrize("error_code", ["context_length_exceeded", "other_error"])
def test_claude_bad_request(monkeypatch, error_code):
    """
    Test that if litellm.completion raises BadRequestError,
    the call method handles it accordingly.
    """
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    monkeypatch.setattr(AnthropicModel, "check_api_key", dummy_check_api_key)
    monkeypatch.setattr(AnthropicModel, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # Create a dummy BadRequestError with required args.
    err = BadRequestError("error", response=DummyResponseObject(), body={})
    err.code = error_code

    monkeypatch.setattr(
        "litellm.completion", lambda **kwargs: (_ for _ in ()).throw(err)
    )

    model = Claude3Opus()
    messages = [{"role": "user", "content": "Test"}]

    with pytest.raises(BadRequestError):
        model.call(messages, temperature=1.0)
    print(f"BadRequestError test passed for error code '{error_code}'.")
