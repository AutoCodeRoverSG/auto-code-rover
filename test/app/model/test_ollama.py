import time
import sys
import pytest
from tenacity import RetryError
from app.model.ollama import OllamaModel, Llama3_8B, Llama3_70B
from app.model import common

from test.pytest_utils import *

# Assume these are imported from test/pytest_utils:
# DummyThreadCost, dummy_check_api_key, dummy_sleep, dummy_sys_exit, SysExitException


# --- Dummy Response for Ollama Models ---
def DummyOllamaResponse(content="Test response"):
    """Return a dummy response mimicking the Ollama API."""
    return {"message": {"content": content}}


# --- Dummy Check Functions for Ollama ---
def dummy_check_api_key_ollama(self):
    print("dummy_check_api_key_ollama called")
    return "No key required for local models."


def dummy_send_empty_request(self):
    print("dummy_send_empty_request called")
    # Do nothing for testing.


# -------------------- Ollama Model Call Tests --------------------

# Dictionary of Ollama models to test.
ollama_models = {
    "Llama3_8B": Llama3_8B,
    "Llama3_70B": Llama3_70B,
}


@pytest.mark.parametrize(
    "model_class", ollama_models.values(), ids=ollama_models.keys()
)
def test_ollama_model_call(monkeypatch, model_class):
    """
    Test the call method for Ollama models.
      - For response_format "text": returns dummy response.
      - For response_format "json_object": verifies that an extra JSON instruction is appended.
    """
    # Clear the singleton cache.
    model_class._instances = {}
    # Patch timeout_decorator.timeout to bypass timeouts.
    monkeypatch.setattr("timeout_decorator.timeout", lambda t: (lambda f: f))
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    monkeypatch.setattr(OllamaModel, "check_api_key", dummy_check_api_key_ollama)
    monkeypatch.setattr(OllamaModel, "send_empty_request", dummy_send_empty_request)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # --- Case 1: response_format = "text" ---
    monkeypatch.setattr("ollama.chat", lambda **kwargs: DummyOllamaResponse())
    model = model_class()
    messages = [{"role": "user", "content": "Hello"}]
    result = model.call(messages, response_format="text")
    # Ollama call returns a tuple: (content, 0, 0, 0)
    content, cost, input_tokens, output_tokens = result
    assert content == "Test response"
    assert cost == 0
    assert input_tokens == 0
    assert output_tokens == 0

    # --- Case 2: response_format = "json_object" ---
    # We'll capture the messages passed to ollama.chat.
    captured_messages = []

    def dummy_ollama_chat(**kwargs):
        captured_messages.append(kwargs.get("messages"))
        return DummyOllamaResponse()

    monkeypatch.setattr("ollama.chat", dummy_ollama_chat)
    messages = [{"role": "user", "content": "Hello"}]
    result = model.call(messages, response_format="json_object")
    content_json, cost_json, input_tokens_json, output_tokens_json = result
    # In the json_object branch, the method appends an extra instruction to messages:
    # {"role": "user", "content": "Stop your response after a valid json is generated."}
    assert messages[-1] == {
        "role": "user",
        "content": "Stop your response after a valid json is generated.",
    }
    # The dummy response always returns "Test response".
    assert content_json == "Test response"
    print(f"Ollama model {model_class.__name__} passed call tests.")
