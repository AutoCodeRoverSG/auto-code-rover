import time
import sys
import os
import pytest
from unittest.mock import MagicMock
from tenacity import RetryError
from app.model.azure import *
from app.model import common
from openai import BadRequestError
from test.pytest_utils import *


# --- Dummy Response for Azure Model Call ---
def DummyAzureResponse(
    content="Test response", input_tokens=1, output_tokens=2, tool_calls=None
):
    """Returns a dummy response object mimicking a ChatCompletion."""
    dummy = MagicMock()
    # Create a dummy usage object with required token counts.
    Usage = type(
        "Usage", (), {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
    )
    dummy.usage = Usage()
    # Simulate one choice with a message.
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    dummy.choices = [MagicMock(message=msg)]
    return dummy


# --- Dummy Check Functions & Thread Cost ---
def dummy_check_api_key_azure(self):
    print("dummy_check_api_key_azure called")
    return "dummy-azure-key"


def dummy_check_endpoint_url(self):
    print("dummy_check_endpoint_url called")
    return "https://dummy.endpoint"


# --- Dummy Azure Client ---
class DummyAzureCompletions:
    last_kwargs = None

    def create(self, *args, **kwargs):
        print("DummyAzureCompletions.create called")
        DummyAzureCompletions.last_kwargs = kwargs
        return DummyAzureResponse()


class DummyAzureClientChat:
    completions = DummyAzureCompletions()


class DummyAzureClient:
    chat = DummyAzureClientChat()


# ===================== Azure Model Tests =====================


# --- Test Setup and API Key / Endpoint Checks ---
def test_azure_setup(monkeypatch):
    monkeypatch.setattr(AzureGpt_o1mini, "check_api_key", dummy_check_api_key_azure)
    monkeypatch.setattr(AzureGpt_o1mini, "check_endpoint_url", dummy_check_endpoint_url)
    model = AzureGpt_o1mini()
    model.setup()
    # Check that client is set.
    assert model.client is not None


def test_azure_check_api_key_success(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda key: "dummy-azure-key")
    model = AzureGpt_o1mini()
    key = model.check_api_key()
    assert key == "dummy-azure-key"


def test_azure_check_api_key_failure(monkeypatch, capsys):
    monkeypatch.setattr(os, "getenv", lambda key: "")
    monkeypatch.setattr(sys, "exit", dummy_sys_exit)
    model = AzureGpt_o1mini()
    with pytest.raises(SysExitException):
        model.check_api_key()
    captured = capsys.readouterr().out
    assert "Please set the AZURE_OPENAI_KEY env var" in captured


def test_azure_check_endpoint_url_failure(monkeypatch, capsys):
    monkeypatch.setattr(os, "getenv", lambda key: "")
    monkeypatch.setattr(sys, "exit", dummy_sys_exit)
    model = AzureGpt_o1mini()
    with pytest.raises(SysExitException):
        model.check_endpoint_url()
    captured = capsys.readouterr().out
    assert "Please set the ENDPOINT_URL env var" in captured


# --- Test extract_resp_content ---
def test_extract_resp_content_azure():
    model = AzureGpt_o1mini()

    class DummyMessage:
        def __init__(self, content):
            self.content = content
            self.tool_calls = None

    msg = DummyMessage("Hello")
    assert model.extract_resp_content(msg) == "Hello"
    msg_none = DummyMessage(None)
    assert model.extract_resp_content(msg_none) == ""


# --- Define Dictionary of Azure Models to Test ---
azure_models = {
    "AzureGpt_o1mini": AzureGpt_o1mini,
    "AzureGpt4o": AzureGpt4o,
    "AzureGpt35_Turbo": AzureGpt35_Turbo,
    "AzureGpt35_Turbo16k": AzureGpt35_Turbo16k,
    "AzureGpt4": AzureGpt4,
}


# --- Test Normal Call Flow for Azure Models ---
@pytest.mark.parametrize("model_class", azure_models.values(), ids=azure_models.keys())
def test_azure_model_call(monkeypatch, model_class):
    """
    Test the normal call flow of Azure models for both response formats.
    """
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    monkeypatch.setattr(AzureOpenaiModel, "check_api_key", dummy_check_api_key_azure)
    monkeypatch.setattr(
        AzureOpenaiModel, "check_endpoint_url", dummy_check_endpoint_url
    )
    monkeypatch.setattr(AzureOpenaiModel, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())
    # Set the dummy Azure client.
    # monkeypatch.setattr(AzureOpenaiModel, "client", DummyAzureClient())

    # ----- Case 1: response_format = "text" -----
    monkeypatch.setattr(DummyAzureClientChat, "completions", DummyAzureCompletions())
    model = model_class()
    model.client = DummyAzureClient()
    messages = [{"role": "user", "content": "Hello"}]
    result = model.call(messages, response_format="text")
    content, raw_tool_calls, func_call_intents, cost, input_tokens, output_tokens = (
        result
    )
    assert content == "Test response"
    assert cost == 0.5
    assert input_tokens == 1
    assert output_tokens == 2


# --- Test BadRequestError Handling for Azure ---
@pytest.mark.parametrize("error_code", ["context_length_exceeded", "other_error"])
def test_azure_bad_request(monkeypatch, error_code):
    """
    Test that if the Azure client's completions.create raises BadRequestError,
    the call method handles it accordingly.
    """
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)
    monkeypatch.setattr(AzureOpenaiModel, "check_api_key", dummy_check_api_key_azure)
    monkeypatch.setattr(
        AzureOpenaiModel, "check_endpoint_url", dummy_check_endpoint_url
    )
    monkeypatch.setattr(AzureOpenaiModel, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # Create a dummy BadRequestError with required args.
    err = BadRequestError("error", response=DummyResponseObject(), body={})
    err.code = error_code

    def dummy_error(**kwargs):
        return (_ for _ in ()).throw(err)

    monkeypatch.setattr(DummyAzureClientChat.completions, "create", dummy_error)

    model = AzureGpt_o1mini()
    model.client = DummyAzureClient()
    messages = [{"role": "user", "content": "Test"}]
    with pytest.raises(RetryError):
        model.call(messages, response_format="text")
    print(f"Azure BadRequestError test passed for error code '{error_code}'.")
