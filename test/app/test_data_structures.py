import json
import pytest
from app.data_structures import FunctionCallIntent, OpenaiFunction


# Define a dummy OpenaiFunction for testing.
class DummyOpenaiFunction:
    def __init__(self, arguments, name):
        self.arguments = arguments
        self.name = name

    def __eq__(self, other):
        return (
            isinstance(other, DummyOpenaiFunction)
            and self.arguments == other.arguments
            and self.name == other.name
        )

    def __str__(self):
        return f"DummyOpenaiFunction(name={self.name}, arguments={self.arguments})"


# Automatically replace OpenaiFunction in the module with DummyOpenaiFunction for tests.
@pytest.fixture(autouse=True)
def use_dummy_openai_function(monkeypatch):
    monkeypatch.setattr(
        "app.data_structures.OpenaiFunction",
        DummyOpenaiFunction,
    )


def test_function_call_intent_default():
    func_name = "test_func"
    arguments = {"arg1": "value1", "arg2": "value2"}
    # When no openai_func is provided, it should create one using DummyOpenaiFunction.
    intent = FunctionCallIntent(func_name, arguments, None)
    # Check that func_name is set.
    assert intent.func_name == func_name
    # Check that arg_values equals the provided arguments.
    assert intent.arg_values == arguments
    # Check that openai_func is created and has the proper values.
    assert isinstance(intent.openai_func, DummyOpenaiFunction)
    expected_args = json.dumps(arguments)
    assert intent.openai_func.arguments == expected_args
    assert intent.openai_func.name == func_name


def test_function_call_intent_with_openai_func():
    func_name = "another_func"
    arguments = {"x": "1"}
    # Create a dummy openai function.
    dummy_func = DummyOpenaiFunction(
        arguments=json.dumps({"x": "override"}), name="dummy"
    )
    intent = FunctionCallIntent(func_name, arguments, dummy_func)
    # The provided openai_func should be used.
    assert intent.openai_func == dummy_func
    # And arg_values should still reflect the passed arguments.
    assert intent.arg_values == arguments


def test_to_dict():
    func_name = "func_to_dict"
    arguments = {"a": "b"}
    intent = FunctionCallIntent(func_name, arguments, None)
    result = intent.to_dict()
    expected = {"func_name": func_name, "arguments": arguments}
    assert result == expected


def test_to_dict_with_result():
    func_name = "func_with_result"
    arguments = {"key": "val"}
    intent = FunctionCallIntent(func_name, arguments, None)
    result_true = intent.to_dict_with_result(True)
    result_false = intent.to_dict_with_result(False)
    expected_true = {"func_name": func_name, "arguments": arguments, "call_ok": True}
    expected_false = {"func_name": func_name, "arguments": arguments, "call_ok": False}
    assert result_true == expected_true
    assert result_false == expected_false


def test_str_method():
    func_name = "str_func"
    arguments = {"param": "123"}
    intent = FunctionCallIntent(func_name, arguments, None)
    s = str(intent)
    # The string representation should include the function name and the arguments.
    assert func_name in s
    assert str(arguments) in s
