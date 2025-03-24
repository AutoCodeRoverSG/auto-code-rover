import json
import pytest
from app.data_structures import MessageThread
from app.model import common
from app.agents.agent_select import run
from test.pytest_utils import *  # Import any helper utilities


def test_run_pr_review():
    """
    Test the run function for pull request review.

    The dummy model simulates:
      - Two initial analysis calls.
      - Three loop iterations returning JSON responses indicating patch number 2.

    With two patches provided, the chosen patch should yield index 1 (zero-indexed).
    """
    responses = [
        "dummy analysis 1",  # Analysis: root cause.
        "dummy analysis 2",  # Analysis: resolution.
        '{"patch_number": 2, "reason": "patch2 is minimal"}',
        '{"patch_number": 2, "reason": "patch2 is minimal"}',
        '{"patch_number": 2, "reason": "patch2 is minimal"}',
    ]
    dummy_model = DummyModel(responses)
    # Replace MODEL_HUB entry.
    common.MODEL_HUB = {"gpt-4-0125-preview": dummy_model}

    issue_statement = "This is a test issue"
    patch_contents = ["patch content 1", "patch content 2"]

    index, reason, prefix_thread = run(issue_statement, patch_contents)

    # Expect patch number 2 => index 1 (zero-indexed)
    assert index == 1, f"Expected index 1, got {index}"
    assert reason == "patch2 is minimal", f"Unexpected reason: {reason}"

    # Verify that prefix_thread is a MessageThread and contains at least one JSON response.
    assert isinstance(prefix_thread, MessageThread)
    json_found = any(
        '{"patch_number": 2' in msg.get("content", "")
        for msg in prefix_thread.messages
        if msg.get("role") in ("model", "assistant")
    )
    assert (
        json_found
    ), "Expected a model/assistant message containing the JSON response was not found."
