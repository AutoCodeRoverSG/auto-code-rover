import json
import sys
import time
import os
import pytest

from app.model.gpt import *
from app.data_structures import FunctionCallIntent
from app.model import common
from openai import BadRequestError
from tenacity import RetryError
from test.pytest_utils import *


# Dummy tool call classes for testing extract_resp_func_calls.
class DummyOpenaiFunction:
    def __init__(self, name="dummy_function", arguments='{"arg": "value"}'):
        self.name = name
        self.arguments = arguments


class DummyToolCall:
    def __init__(self, name="dummy_function", arguments='{"arg": "value"}'):
        self.function = DummyOpenaiFunction(name, arguments)


class DummyToolCallEmpty:
    def __init__(self, name="dummy_function"):
        # Empty string branch.
        self.function = DummyOpenaiFunction(name, "")


class DummyToolCallInvalid:
    def __init__(self, name="dummy_function"):
        # Invalid JSON branch.
        self.function = DummyOpenaiFunction(name, "{invalid_json}")


# Dummy OpenAI client to use in testing.
class DummyOpenAI:
    def __init__(self, api_key):
        self.api_key = api_key


# -- Tests for setup ---
def test_setup_initializes_client(monkeypatch):
    # Patch check_api_key to return a dummy key.
    monkeypatch.setattr(Gpt_o1, "check_api_key", dummy_check_api_key)
    # Patch the OpenAI constructor to return a DummyOpenAI instance.
    monkeypatch.setattr("app.model.gpt.OpenAI", lambda api_key: DummyOpenAI(api_key))

    # Instantiate a model.
    model = Gpt_o1()
    # Ensure client is initially None.
    model.client = None

    # Call setup.
    model.setup()

    # Verify that client was set using our dummy OpenAI.
    assert model.client is not None
    assert isinstance(model.client, DummyOpenAI)
    assert model.client.api_key == "dummy-key"


def test_setup_already_initialized(monkeypatch):
    # Patch check_api_key (should not be called if client is already set).
    monkeypatch.setattr(Gpt_o1, "check_api_key", dummy_check_api_key)
    # Create a dummy client.
    dummy_client = DummyOpenAI("pre-set-key")

    # Instantiate a model and set the client.
    model = Gpt_o1()
    model.client = dummy_client

    # Call setup again.
    model.setup()

    # Verify that client remains unchanged.
    assert model.client is dummy_client


# --- Test edge cases for one model


def test_singleton_behavior(monkeypatch):
    # Ensure that __new__ returns the same instance when __init__ is skipped.
    monkeypatch.setattr(OpenaiModel, "check_api_key", dummy_check_api_key)
    # Create an instance and attach an attribute.
    model1 = Gpt_o1()
    model1.some_attr = "initial"
    # Create another instance.
    model2 = Gpt_o1()
    # They should be the same object.
    assert model1 is model2
    # __init__ should not reinitialize; the attribute remains.
    assert hasattr(model2, "some_attr")
    assert model2.some_attr == "initial"


def test_check_api_key_success(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda key: "dummy-key")
    model = Gpt_o1()
    key = model.check_api_key()
    assert key == "dummy-key"


def test_check_api_key_failure(monkeypatch, capsys):
    # Simulate missing OPENAI_KEY.
    monkeypatch.setattr(os, "getenv", lambda key: "")
    monkeypatch.setattr(sys, "exit", dummy_sys_exit)
    model = Gpt_o1()
    with pytest.raises(SysExitException):
        model.check_api_key()
    captured = capsys.readouterr().out
    assert "Please set the OPENAI_KEY env var" in captured


def test_extract_resp_content(monkeypatch):
    model = Gpt_o1()
    # Test when content exists.
    msg = DummyMessage(content="Hello")
    assert model.extract_resp_content(msg) == "Hello"
    # Test when content is None.
    msg_none = DummyMessage(content=None)
    assert model.extract_resp_content(msg_none) == ""


def test_extract_resp_func_calls(monkeypatch):
    model = Gpt_o1()
    # When tool_calls is None.
    msg = DummyMessage(tool_calls=None)
    assert model.extract_resp_func_calls(msg) == []
    # When arguments is an empty string.
    dummy_call_empty = DummyToolCallEmpty()
    msg_empty = DummyMessage(tool_calls=[dummy_call_empty])
    func_calls = model.extract_resp_func_calls(msg_empty)
    assert len(func_calls) == 1
    assert func_calls[0].func_name == "dummy_function"
    assert func_calls[0].arg_values == {}
    # When arguments is invalid JSON.
    dummy_call_invalid = DummyToolCallInvalid()
    msg_invalid = DummyMessage(tool_calls=[dummy_call_invalid])
    func_calls = model.extract_resp_func_calls(msg_invalid)
    assert len(func_calls) == 1
    assert func_calls[0].func_name == "dummy_function"
    assert func_calls[0].arg_values == {}
    # When arguments is valid.
    dummy_call = DummyToolCall()
    msg_valid = DummyMessage(tool_calls=[dummy_call])
    func_calls = model.extract_resp_func_calls(msg_valid)
    assert len(func_calls) == 1
    assert func_calls[0].func_name == "dummy_function"
    assert func_calls[0].arg_values == {"arg": "value"}


# --- Parametrized Test Over All Models ---


@pytest.mark.parametrize(
    "model_class, expected_name",
    [
        ("Gpt_o1mini", "o1-mini"),
        ("Gpt_o1", "o1-2024-12-17"),
        ("Gpt4o_20240806", "gpt-4o-2024-08-06"),
        ("Gpt4o_20240513", "gpt-4o-2024-05-13"),
        ("Gpt4_Turbo20240409", "gpt-4-turbo-2024-04-09"),
        ("Gpt4_0125Preview", "gpt-4-0125-preview"),
        ("Gpt4_1106Preview", "gpt-4-1106-preview"),
        ("Gpt35_Turbo0125", "gpt-3.5-turbo-0125"),
        ("Gpt35_Turbo1106", "gpt-3.5-turbo-1106"),
        ("Gpt35_Turbo16k_0613", "gpt-3.5-turbo-16k-0613"),
        ("Gpt35_Turbo0613", "gpt-3.5-turbo-0613"),
        ("Gpt4_0613", "gpt-4-0613"),
        ("Gpt4o_mini_20240718", "gpt-4o-mini-2024-07-18"),
    ],
)
def test_openai_model_call(monkeypatch, model_class, expected_name):
    # Dynamically import the model class from the gpt module.
    from app.model import gpt

    model_cls = getattr(gpt, model_class)

    # Patch necessary methods.
    monkeypatch.setattr(model_cls, "check_api_key", dummy_check_api_key)
    monkeypatch.setattr(model_cls, "calc_cost", lambda self, inp, out: 0.5)
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # Instantiate the model and set the dummy client.
    model = model_cls()
    model.client = DummyClient()

    # Prepare a dummy messages list to simulate a user call.
    messages = [{"role": "user", "content": "Hello"}]

    # Call the model's "call" method.
    result = model.call(messages)
    content, raw_tool_calls, func_call_intents, cost, input_tokens, output_tokens = (
        result
    )

    # Verify the model's name and basic call flow.
    assert model.name == expected_name
    assert content == "Test response"
    assert raw_tool_calls is None  # Because DummyMessage.tool_calls was None.
    assert func_call_intents == []  # No tool calls provided.
    assert cost == 0.5
    assert input_tokens == 1
    assert output_tokens == 2

    # Check that our dummy thread cost was updated.
    assert common.thread_cost.process_cost == 0.5
    assert common.thread_cost.process_input_tokens == 1
    assert common.thread_cost.process_output_tokens == 2

    # Test extract_resp_content separately.
    dummy_msg = DummyMessage()
    extracted_content = model.extract_resp_content(dummy_msg)
    assert extracted_content == "Test response"

    # Test extract_resp_func_calls by simulating a tool call.
    dummy_msg.tool_calls = [DummyToolCall()]
    func_calls = model.extract_resp_func_calls(dummy_msg)
    assert len(func_calls) == 1
    # Check that the function call intent is correctly extracted.
    assert func_calls[0].func_name == "dummy_function"
    assert func_calls[0].arg_values == {"arg": "value"}


def test_call_single_tool_branch(monkeypatch):
    # Patch check_api_key to return a dummy key.
    monkeypatch.setattr(OpenaiModel, "check_api_key", dummy_check_api_key)
    # Patch calc_cost to return a fixed cost.
    monkeypatch.setattr(OpenaiModel, "calc_cost", lambda self, inp, out: 0.5)
    # Patch common.thread_cost with our dummy instance.
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # Instantiate a model (using Gpt_o1 as an example) and set the dummy client.
    from app.model.gpt import Gpt_o1  # import within test to avoid conflicts

    model = Gpt_o1()
    model.client = DummyClient()

    # Prepare a dummy messages list to simulate a user call.
    messages = [{"role": "user", "content": "Hello"}]
    # Prepare tools with exactly one tool.
    tools = [{"function": {"name": "dummy_tool"}}]

    # Call the model's "call" method.
    result = model.call(messages, tools=tools, temperature=1.0)

    # Access the kwargs passed to DummyCompletions.create.
    kw = DummyCompletions.last_kwargs
    # Assert that tool_choice was added.
    assert "tool_choice" in kw
    # Check that the tool_choice has the expected structure.
    assert kw["tool_choice"]["type"] == "function"
    assert kw["tool_choice"]["function"]["name"] == "dummy_tool"


def test_call_bad_request(monkeypatch):
    # Do not patch log_and_print so that the actual lines in the except block execute.
    # Disable sleep functions so that no real delays occur.
    monkeypatch.setattr("tenacity.sleep", dummy_sleep)
    monkeypatch.setattr(time, "sleep", dummy_sleep)

    # Patch check_api_key to return a dummy key.
    monkeypatch.setattr(OpenaiModel, "check_api_key", dummy_check_api_key)
    # Patch calc_cost to return a fixed cost.
    monkeypatch.setattr(OpenaiModel, "calc_cost", lambda self, inp, out: 0.5)
    # Replace common.thread_cost with our dummy instance.
    monkeypatch.setattr(common, "thread_cost", DummyThreadCost())

    # Create a dummy client that always raises BadRequestError.
    model = Gpt_o1()
    model.client = DummyBadRequestClient("context_length_exceeded")

    messages = [{"role": "user", "content": "Hello"}]

    print("Calling model.call with messages:", messages)
    with pytest.raises(RetryError) as exc_info:
        model.call(messages, temperature=1.0)

    # Extract the last exception from the RetryError chain.
    last_exc = exc_info.value.last_attempt.exception()
    print("Final exception from last attempt:", last_exc)

    # Walk the cause chain to see if BadRequestError is present.
    chain = extract_exception_chain(last_exc)
    for i, e in enumerate(chain):
        print(
            f"Exception in chain [{i}]: type={type(e)}, message={getattr(e, 'message', str(e))}, code={getattr(e, 'code', None)}"
        )

    # Assert that one exception in the chain is a BadRequestError with the expected code.
    found = any(
        isinstance(e, BadRequestError)
        and getattr(e, "code", None) == "context_length_exceeded"
        for e in chain
    )
    assert found, "BadRequestError with expected code not found in exception chain."

    # Other tests with different error codes.
    model.client = DummyBadRequestClient("some_other_code")
    messages = [{"role": "user", "content": "Hello"}]

    print("Calling model.call with messages:", messages)
    with pytest.raises(RetryError) as exc_info:
        model.call(messages, temperature=1.0)

    # Extract the last exception from the RetryError chain.
    last_exc = exc_info.value.last_attempt.exception()
    print("Final exception from last attempt:", last_exc)

    # Walk the cause chain to see if BadRequestError is present.
    chain = extract_exception_chain(last_exc)
    for i, e in enumerate(chain):
        print(
            f"Exception in chain [{i}]: type={type(e)}, message={getattr(e, 'message', str(e))}, code={getattr(e, 'code', None)}"
        )

    # Assert that one exception in the chain is a BadRequestError with the expected code.
    found = any(
        isinstance(e, BadRequestError) and getattr(e, "code", None) == "some_other_code"
        for e in chain
    )
    assert found, "BadRequestError with expected code not found in exception chain."
