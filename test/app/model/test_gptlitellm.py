import os
import time
import pytest
from tenacity import RetryError
from app.model import common

from app.model.gptlitellm import *
from litellm.utils import ModelResponse, Choices, Message


# --- Dummy Utilities ---
def DummyLiteLLMResponse(
    content="Test response", input_tokens=1, output_tokens=2, tool_calls=None
):
    """
    Create a dummy response as a valid ModelResponse instance using parse_obj.
    """
    data = {
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        "choices": [{"message": {"content": content, "tool_calls": tool_calls}}],
    }
    return ModelResponse.parse_obj(data)


def dummy_check_api_key_litellm(self):
    print("dummy_check_api_key_litellm called")
    return "dummy-key"


class DummyThreadCost:
    process_cost = 0.0
    process_input_tokens = 0
    process_output_tokens = 0


def dummy_sleep(seconds):
    print(f"dummy_sleep called with {seconds} seconds (disabled)")
    return None


# -------------------- OpenaiLiteLLMModel Call Tests --------------------

# Dictionary of LiteLLM models to test.
lite_llm_models = {
    "Gpt4o_20240513LiteLLM": Gpt4o_20240513LiteLLM,
    "Gpt4_Turbo20240409LiteLLM": Gpt4_Turbo20240409LiteLLM,
    "Gpt4_0125PreviewLiteLLM": Gpt4_0125PreviewLiteLLM,
    "Gpt4_1106PreviewLiteLLM": Gpt4_1106PreviewLiteLLM,
    "Gpt35_Turbo0125LiteLLM": Gpt35_Turbo0125LiteLLM,
    "Gpt35_Turbo1106LiteLLM": Gpt35_Turbo1106LiteLLM,
    "Gpt35_Turbo16k_0613LiteLLM": Gpt35_Turbo16k_0613LiteLLM,
    "Gpt35_Turbo0613LiteLLM": Gpt35_Turbo0613LiteLLM,
    "Gpt4_0613LiteLLM": Gpt4_0613LiteLLM,
}


@pytest.mark.parametrize(
    "model_class", lite_llm_models.values(), ids=lite_llm_models.keys()
)
def test_openai_litellm_model_call(monkeypatch, model_class):
    """
    Test the normal call flow of OpenaiLiteLLM models.
      - For response_format "text": it should return the dummy response.
      - For response_format "json_object": it should prefill the last message with "{".
    """
    # Clear singleton cache.
    model_class._instances = {}
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    monkeypatch.setattr(model_class, "check_api_key", dummy_check_api_key_litellm)
    monkeypatch.setattr(model_class, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # --- Case 1: response_format = "text" ---
    monkeypatch.setattr("litellm.completion", lambda **kwargs: DummyLiteLLMResponse())
    model = model_class()
    messages = [{"role": "user", "content": "Hello"}]
    result = model.call(messages, response_format="text")
    # OpenaiLiteLLMModel.call returns: content, cost, input_tokens, output_tokens
    content, cost, input_tokens, output_tokens = result
    assert content == "Test response"
    assert cost == 0.5
    assert input_tokens == 1
    assert output_tokens == 2

    # --- Case 2: response_format = "json_object" ---
    # When "json_object" is used, the call method appends an assistant message with prefill "{"
    # and then later prepends the prefill if needed.
    captured_messages = []

    def dummy_completion(**kwargs):
        captured_messages.append(kwargs.get("messages"))
        return DummyLiteLLMResponse()

    monkeypatch.setattr("litellm.completion", dummy_completion)
    messages = [{"role": "user", "content": "Hello"}]
    result = model.call(messages, response_format="json_object")
    content_json, cost_json, input_tokens_json, output_tokens_json = result
    prefill = "{"
    # Because the source appends an assistant message with prefill and then, if the content doesn't start with it,
    # prepends it, the final content should be prefill + dummy content.
    expected = prefill + "Test response"
    assert content_json == expected
    print(f"OpenaiLiteLLM model {model_class.__name__} passed call tests.")
