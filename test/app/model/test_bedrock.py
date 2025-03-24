import time
import sys
import os
import pytest
from tenacity import RetryError
from app.model.bedrock import (
    BedrockModel,
    AmazonNovaLitev1,
    AmazonNovaProv1,
    AmazonNovaMicrov1,
    AnthropicClaude2,
    AnthropicClaude3Opus,
    AnthropicClaude3Sonnet,
    AnthropicClaude35Sonnet,
    AnthropicClaude3Haiku,
)
from app.model import common
from openai import BadRequestError
from litellm.utils import ModelResponse, Choices, Message
from app.log import log_and_print
from test.pytest_utils import *


# --- Updated Dummy Response for Bedrock Models using parse_obj ---
def DummyBedrockResponse(
    content="Test response", input_tokens=1, output_tokens=2, tool_calls=None
):
    """
    Create a dummy response that is a valid instance of ModelResponse.
    """
    data = {
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        "choices": [{"message": {"content": content, "tool_calls": tool_calls}}],
    }
    return ModelResponse.parse_obj(data)


# -------------------- Bedrock Model Call Tests --------------------

# Dictionary of Bedrock models to test.
bedrock_models = {
    "AmazonNovaLitev1": AmazonNovaLitev1,
    "AmazonNovaProv1": AmazonNovaProv1,
    "AmazonNovaMicrov1": AmazonNovaMicrov1,
    "AnthropicClaude2": AnthropicClaude2,
    "AnthropicClaude3Opus": AnthropicClaude3Opus,
    "AnthropicClaude3Sonnet": AnthropicClaude3Sonnet,
    "AnthropicClaude35Sonnet": AnthropicClaude35Sonnet,
    "AnthropicClaude3Haiku": AnthropicClaude3Haiku,
}


@pytest.mark.parametrize(
    "model_class", bedrock_models.values(), ids=bedrock_models.keys()
)
def test_bedrock_model_call(monkeypatch, model_class):
    """
    Test the normal call flow of Bedrock models.
      - For response_format "text": returns a dummy response.
      - For response_format "json_object": verifies that if the model’s _model_provider
        starts with "bedrock/anthropic", the assistant message is prefilled with "{".
    """
    # Clear the singleton cache.
    model_class._instances = {}
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    monkeypatch.setattr(BedrockModel, "check_api_key", dummy_check_api_key)
    monkeypatch.setattr(BedrockModel, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # --- Case 1: response_format = "text" ---
    monkeypatch.setattr("litellm.completion", lambda **kwargs: DummyBedrockResponse())
    model = model_class()
    messages = [{"role": "user", "content": "Hello"}]
    result = model.call(messages, response_format="text")
    # Bedrock call returns: content, cost, input_tokens, output_tokens
    content, cost, input_tokens, output_tokens = result
    assert content == "Test response"
    assert cost == 0.5
    assert input_tokens == 1
    assert output_tokens == 2

    # --- Case 2: response_format = "json_object" ---
    captured_messages = []

    def dummy_completion(**kwargs):
        captured_messages.append(kwargs.get("messages"))
        return DummyBedrockResponse()

    monkeypatch.setattr("litellm.completion", dummy_completion)
    messages = [{"role": "user", "content": "Hello"}]
    result = model.call(messages, response_format="json_object")
    content_json, cost_json, input_tokens_json, output_tokens_json = result
    # Determine prefill: if _model_provider starts with "bedrock/anthropic", a prefill "{" is used.
    prefill = "{" if model._model_provider.startswith("bedrock/anthropic") else ""
    if prefill:
        # The call method appends an assistant message with prefill and then later prepends it if needed.
        expected = prefill + "Test response"
        assert content_json == expected
    else:
        assert content_json == "Test response"
    print(f"Bedrock model {model_class.__name__} passed call tests.")


# --- Introduce a problematic statement inside the test files to check if SonarQube catches issues inside test code
# --- This statement is not a problem in the actual codebase.
# --- Returning hash as a string to check if SonarQube catches this issue.


class SomeClass:
    def __init__(self):
        self.foo = "foo"
        self.bar = "bar"

    def __hash__(self):
        return "hash((self.foo, self.bar))"
