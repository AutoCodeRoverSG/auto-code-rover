import time
import pytest
from unittest.mock import MagicMock
from tenacity import RetryError
from app.model.claude import AnthropicModel, Claude3Opus, Claude3Sonnet, Claude3Haiku, Claude3_5Sonnet, Claude3_5SonnetNew
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
    Usage = type("Usage", (), {"prompt_tokens": input_tokens, "completion_tokens": output_tokens})
    dummy.usage = Usage()
    return dummy

# --- Define dictionary of Claude models to test ---
claude_models = {
    "Claude3Opus": Claude3Opus,
    "Claude3Sonnet": Claude3Sonnet,
    "Claude3Haiku": Claude3Haiku,
    "Claude3_5Sonnet": Claude3_5Sonnet,
    "Claude3_5SonnetNew": Claude3_5SonnetNew,
}

@pytest.mark.parametrize("model_class", claude_models.values(), ids=claude_models.keys())
def test_anthropic_model_call(monkeypatch, model_class):
    """
    Test the normal call flow of Anthropic models.
    """
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    monkeypatch.setattr(AnthropicModel, "check_api_key", dummy_check_api_key)
    monkeypatch.setattr(AnthropicModel, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # Patch litellm.completion to return our dummy response.
    monkeypatch.setattr("litellm.completion", lambda **kwargs: DummyLiteLLMResponse())

    model = model_class()
    messages = [{"role": "user", "content": "Hello"}]
    # Use response_format "text" to avoid modifying the message.
    content, cost, input_tokens, output_tokens = model.call(messages, temperature=1.0)
    assert content == "Test response"
    assert cost == 0.5
    assert input_tokens == 1
    assert output_tokens == 2
    print(f"Test passed for {model_class.__name__}")

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

    monkeypatch.setattr("litellm.completion", lambda **kwargs: (_ for _ in ()).throw(err))

    model = Claude3Opus()
    messages = [{"role": "user", "content": "Test"}]

    with pytest.raises(BadRequestError):
        model.call(messages, temperature=1.0)
    print(f"BadRequestError test passed for error code '{error_code}'.")
