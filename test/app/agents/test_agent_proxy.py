import json
import inspect
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest
from app.agents.agent_proxy import (
    run,
    run_with_retries,
    is_valid_response,
    PROXY_PROMPT,
)
from app.data_structures import MessageThread
from app.post_process import ExtractStatus, is_valid_json
from app.search.search_backend import SearchBackend
from app.utils import parse_function_invocation
from test.pytest_utils import *  # Import any shared test utilities


# Fixture to restore common.SELECTED_MODEL after each test
@pytest.fixture(autouse=True)
def restore_selected_model():
    from app.model import common as common_mod

    original = common_mod.__dict__.get("SELECTED_MODEL", None)
    yield
    common_mod.SELECTED_MODEL = original


###############################################################################
# Tests for is_valid_response
###############################################################################
def test_is_valid_response_non_dict():
    """
    When the input data is not a dictionary, the function should return False with a proper message.
    """
    valid, msg = is_valid_response("not a dict")
    assert valid is False
    assert msg == "Json is not a dict"


def test_is_valid_response_empty_dict():
    """
    When neither API_calls nor bug_locations are provided, the function should return an error.
    """
    valid, msg = is_valid_response({})
    assert valid is False
    assert msg == "Both API_calls and bug_locations are empty"


def test_is_valid_response_bug_location_missing_details():
    """
    When bug_locations is provided but its items lack any of 'class', 'method', or 'file',
    the function should return an error.
    """
    data = {"bug_locations": [{}]}
    valid, msg = is_valid_response(data)
    assert valid is False
    assert "Bug location not detailed enough" in msg


def test_is_valid_response_valid_bug_locations():
    """
    When bug_locations has at least one of the required keys, the function should return True.
    """
    data = {"bug_locations": [{"file": "path/to/file"}]}
    valid, msg = is_valid_response(data)
    assert valid is True
    assert msg == "OK"


def test_is_valid_response_invalid_api_calls_type():
    """
    If API_calls is provided but contains a non-string element, return an error.
    """
    data = {"API_calls": [123]}
    valid, msg = is_valid_response(data)
    assert valid is False
    assert msg == "Every API call must be a string"


def test_is_valid_response_invalid_api_call_syntax():
    """
    If an API call string does not conform to the expected syntax (i.e. parse_function_invocation fails),
    the function should return an error.
    """
    data = {"API_calls": ["bad_call"]}
    # Force parse_function_invocation to raise an Exception.
    with patch(
        "app.agents.agent_proxy.parse_function_invocation",
        side_effect=Exception("parse error"),
    ):
        valid, msg = is_valid_response(data)
        assert valid is False
        assert "Every API call must be of form" in msg


def test_is_valid_response_nonexistent_api_call():
    """
    If an API call refers to a function that does not exist in SearchBackend, return an error.
    """
    # Prepare an API call that, when parsed, returns a valid function name but not found.
    api_call = 'search_nonexistent("arg")'
    # Patch parse_function_invocation to return a dummy function name and arguments.
    with patch(
        "app.agents.agent_proxy.parse_function_invocation",
        return_value=("search_nonexistent", ["arg"]),
    ):
        valid, msg = is_valid_response({"API_calls": [api_call]})
        assert valid is False
        assert "calls a non-existent function" in msg


def test_is_valid_response_wrong_number_of_arguments(monkeypatch):
    """
    If an API call has the wrong number of arguments for the target function,
    is_valid_response should return an error message indicating "wrong number of arguments".
    """
    api_call = 'search_method("arg1", "arg2")'
    # Use monkeypatch to override parse_function_invocation to return two arguments.
    monkeypatch.setattr(
        "app.agents.agent_proxy.parse_function_invocation",
        lambda s: ("search_method", ["arg1", "arg2"]),
    )

    # Use monkeypatch to set a dummy search_method on SearchBackend that expects one argument (besides self).
    def dummy_function(self, x):
        return x

    monkeypatch.setattr(SearchBackend, "search_method", dummy_function)

    valid, msg = is_valid_response({"API_calls": [api_call]})
    assert valid is False
    assert "wrong number of arguments" in msg


def test_is_valid_response_valid_api_calls(monkeypatch):
    """
    When API_calls are valid (i.e. correct type, syntax, target function exists, and the correct
    number of arguments are provided), is_valid_response should return True and "OK".
    """
    api_call = 'search_method("arg1")'
    monkeypatch.setattr(
        "app.agents.agent_proxy.parse_function_invocation",
        lambda s: ("search_method", ["arg1"]),
    )

    # Set a dummy search_method on SearchBackend that accepts exactly one argument (besides self).
    def dummy_search_method(self, x):
        return x

    monkeypatch.setattr(SearchBackend, "search_method", dummy_search_method)

    valid, msg = is_valid_response({"API_calls": [api_call]})
    assert valid is True
    assert msg == "OK"


###############################################################################
# Tests for run
###############################################################################
@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
def test_run_returns_expected_thread():
    """
    Test the run function by injecting a dummy model that returns a known JSON string.
    Verify that the returned MessageThread contains the proxy prompt and the user text.
    """
    dummy_response = '{"API_calls": [], "bug_locations": [{"file": "dummy.py"}]}'
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL.call.return_value = (dummy_response,)

    test_text = "Sample raw text input"
    res_text, thread = run(test_text)
    # Check that the response matches dummy_response.
    assert res_text == dummy_response
    # Instead of exact equality, check that the system message contains the expected prompt.
    assert PROXY_PROMPT.strip() in thread.messages[0]["content"]
    # And that the user message equals test_text.
    assert test_text in thread.messages[1]["content"]


###############################################################################
# Tests for run_with_retries
###############################################################################
@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
def test_run_with_retries_eventually_valid():
    """
    Simulate a sequence of responses: first a few invalid JSON responses, then a valid one.
    Verify that run_with_retries returns the valid JSON response and the list of MessageThreads.
    """
    # Define a sequence: first two responses are invalid (e.g., not valid JSON),
    # then one valid JSON.
    invalid_response = "not json"
    valid_data = {"API_calls": ["search_method('arg1')"], "bug_locations": []}
    valid_response = json.dumps(valid_data)
    responses = [invalid_response, invalid_response, valid_response]
    dummy_model = DummyModel(responses)
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL = dummy_model

    test_text = "Input for proxy"
    res_text, threads = run_with_retries(test_text, retries=5)
    # We expect the returned response to be valid.
    assert res_text == valid_response
    # We expect multiple message threads were collected.
    assert len(threads) == 3


@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
def test_run_with_retries_never_valid():
    """
    Simulate that every response is invalid. run_with_retries should then return None for the response.
    """
    responses = ["invalid", "still not json"]
    dummy_model = DummyModel(responses)
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL = dummy_model

    test_text = "Another input"
    res_text, threads = run_with_retries(test_text, retries=2)
    assert res_text is None
    # And threads should have two entries.
    assert len(threads) == 2
