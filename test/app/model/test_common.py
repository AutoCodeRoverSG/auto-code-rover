import os
import sys
import pytest
from app.model.common import (
    Model,
    LiteLLMGeneric,
    register_model,
    get_all_model_names,
    set_model,
    MODEL_TEMP,
    MODEL_HUB,  # assuming MODEL_HUB is defined in common.py
)
from litellm import cost_per_token
from test.pytest_utils import (
    DummyThreadCost,
    dummy_check_api_key,
    dummy_sleep,
    dummy_sys_exit,
    SysExitException,
)
from litellm.utils import ModelResponse, Choices, Message

# --- Test calc_cost and get_overall_exec_stats ---


def test_calc_cost_and_stats(monkeypatch):
    # Create a dummy LiteLLMGeneric instance with known cost rates.
    model = LiteLLMGeneric("dummy-model", 0.1, 0.2)
    # Reset thread_cost (which is a threading.local(), so set attributes on common.thread_cost)
    from app.model import common

    common.thread_cost.process_cost = 0.0
    common.thread_cost.process_input_tokens = 0
    common.thread_cost.process_output_tokens = 0

    # Calculate cost for 10 input tokens and 20 output tokens.
    cost = model.calc_cost(10, 20)
    # Expected: 0.1*10 + 0.2*20 = 1 + 4 = 5
    assert cost == 5.0

    # Update thread_cost and retrieve overall stats.
    common.thread_cost.process_cost += cost
    common.thread_cost.process_input_tokens += 10
    common.thread_cost.process_output_tokens += 20
    stats = model.get_overall_exec_stats()
    assert stats["model"] == "dummy-model"
    assert stats["total_input_tokens"] == 10
    assert stats["total_output_tokens"] == 20
    assert stats["total_tokens"] == 30
    assert stats["total_cost"] == 5.0


# --- Test register_model and get_all_model_names ---


def test_register_and_get_all_model_names():
    from app.model.common import MODEL_HUB, register_model, get_all_model_names

    MODEL_HUB.clear()

    # Create two dummy models by subclassing Model.
    class DummyModel(Model):
        def check_api_key(self) -> str:
            return "dummy"

        def setup(self) -> None:
            pass

        def call(self, messages: list[dict], **kwargs):
            pass

    dummy1 = DummyModel("model1", 0.1, 0.2)
    dummy2 = DummyModel("model2", 0.3, 0.4)
    register_model(dummy1)
    register_model(dummy2)
    names = get_all_model_names()
    assert "model1" in names
    assert "model2" in names


# --- Test set_model with invalid model name ---


def test_set_model_invalid(monkeypatch, capsys):
    from app.model.common import set_model

    monkeypatch.setattr(sys, "exit", dummy_sys_exit)
    with pytest.raises(SysExitException):
        set_model("invalid_model")
    captured = capsys.readouterr().out
    assert "Invalid model name" in captured


# TODO: Add more tests for set_model with valid model names.
