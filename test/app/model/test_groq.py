import os
import time
import sys
import pytest
from tenacity import RetryError
from app.model.groq import (
    GroqModel,
    Llama3_8B,
    Llama3_70B,
    Mixtral_8x7B,
    Gemma_7B,
)
from app.model import common
from openai import BadRequestError
from litellm.utils import ModelResponse, Choices, Message
from app.log import log_and_print
from test.pytest_utils import *


# --- Dummy Response for Groq Models ---
def DummyGroqResponse(
    content="Test response", input_tokens=1, output_tokens=2, tool_calls=None
):
    """
    Create a dummy response that is a valid instance of ModelResponse using parse_obj.
    For Groq models, we simulate a response with fixed usage and one choice.
    """
    data = {
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        "choices": [{"message": {"content": content, "tool_calls": tool_calls}}],
    }
    return ModelResponse.parse_obj(data)


# --- Dummy check function for Groq ---
def dummy_check_api_key_groq(self):
    print("dummy_check_api_key_groq called")
    return "dummy-groq-key"


# -------------------- Groq Model Call Tests --------------------

# Dictionary of Groq models to test.
groq_models = {
    "Llama3_8B": Llama3_8B,
    "Llama3_70B": Llama3_70B,
    "Mixtral_8x7B": Mixtral_8x7B,
    "Gemma_7B": Gemma_7B,
}


@pytest.mark.parametrize("model_class", groq_models.values(), ids=groq_models.keys())
def test_groq_model_call(monkeypatch, model_class):
    """
    Test the normal call flow of Groq models.
      - For response_format "text": the call returns a dummy response.
      - For response_format "json_object": the call appends a prefill ("{")
        to the messages, so that the final content equals "{" + "Test response".
    """
    # Clear the singleton cache.
    model_class._instances = {}
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    monkeypatch.setattr(GroqModel, "check_api_key", dummy_check_api_key_groq)
    monkeypatch.setattr(GroqModel, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # --- Case 1: response_format = "text" ---
    monkeypatch.setattr("litellm.completion", lambda **kwargs: DummyGroqResponse())
    model = model_class()
    messages = [{"role": "user", "content": "Hello"}]
    result = model.call(messages, response_format="text")
    # Groq call returns: content, cost, input_tokens, output_tokens
    content, cost, input_tokens, output_tokens = result
    assert content == "Test response"
    assert cost == 0.5
    assert input_tokens == 1
    assert output_tokens == 2

    # --- Case 2: response_format = "json_object" ---
    captured_messages = []

    def dummy_completion(**kwargs):
        captured_messages.append(kwargs.get("messages"))
        return DummyGroqResponse()

    monkeypatch.setattr("litellm.completion", dummy_completion)
    messages = [{"role": "user", "content": "Hello"}]
    result = model.call(messages, response_format="json_object")
    content_json, cost_json, input_tokens_json, output_tokens_json = result
    # For Groq, the code always sets prefill_content = "{".
    prefill = "{"
    # The call method appends an assistant message with prefill, then later, if the returned content
    # does not start with prefill, it prepends it.
    expected = prefill + "Test response"
    assert content_json == expected
    print(f"Groq model {model_class.__name__} passed call tests.")


# --- Optionally: Test Groq API Key Check Failure ---
def test_groq_check_api_key_failure(monkeypatch, capsys):
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(sys, "exit", dummy_sys_exit)
    model = Llama3_8B()  # any Groq model
    with pytest.raises(SysExitException):
        model.check_api_key()
    captured = capsys.readouterr().out
    assert "Please set the GROQ_API_KEY env var" in captured
