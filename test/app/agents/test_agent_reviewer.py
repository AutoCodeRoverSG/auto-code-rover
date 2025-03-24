import json
import pytest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from app.agents.agent_reviewer import (
    SYSTEM_PROMPT,
    extract_review_result,
    run,
    run_with_retries,
    Review,
    ReviewDecision,
)
from app.data_structures import MessageThread, ReproResult
from app.agents.agent_common import InvalidLLMResponse
from app.model import common
from test.pytest_utils import *  # Import any test helper utilities


###############################################################################
# Tests for extract_review_result
###############################################################################
def test_extract_review_result_valid():
    """
    Verify that extract_review_result returns a Review object when provided
    with valid JSON that includes non-empty advice when decisions are "no".
    """
    # Define the original data dictionary.
    data = {
        "patch-correct": "yes",
        "patch-analysis": "good patch",
        "patch-advice": "",
        "test-correct": "no",
        "test-analysis": "test is inadequate",
        "test-advice": "improve test",
    }
    # Serialize the data into a JSON string.
    json_str = json.dumps(data)
    # Extract a Review instance from the JSON.
    review = extract_review_result(json_str)
    assert review is not None, "Expected a valid Review object, got None"
    # Verify that the instance's to_json() returns the same dictionary.
    assert (
        review.to_json() == data
    ), "The to_json output does not match the original data"
    # Verify individual fields.
    assert review.patch_decision == ReviewDecision.YES
    assert review.test_decision == ReviewDecision.NO
    assert review.patch_analysis == "good patch"
    assert review.test_advice == "improve test"


def test_extract_review_result_invalid_json():
    """
    Verify that extract_review_result returns None when provided with invalid JSON.
    """
    json_str = "not a json"
    review = extract_review_result(json_str)
    assert review is None


def test_extract_review_result_empty_advice():
    """
    Verify that extract_review_result returns None when both patch and test decisions are "no"
    and their corresponding advice fields are empty.
    """
    json_str = json.dumps(
        {
            "patch-correct": "no",
            "patch-analysis": "bad patch",
            "patch-advice": "",
            "test-correct": "no",
            "test-analysis": "bad test",
            "test-advice": "",
        }
    )
    review = extract_review_result(json_str)
    assert review is None


###############################################################################
# Tests for run_with_retries
###############################################################################
@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
def test_run_with_retries_valid_first_attempt():
    """
    Test run_with_retries when the dummy model returns a valid review JSON
    on the first attempt. Verify that the generator yields a valid Review and a
    MessageThread containing the SYSTEM_PROMPT.
    """
    valid_json = json.dumps(
        {
            "patch-correct": "yes",
            "patch-analysis": "patch is correct",
            "patch-advice": "",
            "test-correct": "yes",
            "test-analysis": "test is valid",
            "test-advice": "",
        }
    )
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL.call.return_value = (valid_json,)

    # Dummy inputs for run_with_retries.
    issue_statement = "Test issue"
    test_str = "dummy test"
    patch_str = "dummy patch"
    orig_stdout = "orig stdout"
    orig_stderr = "orig stderr"
    patched_stdout = "patched stdout"
    patched_stderr = "patched stderr"
    gen = run_with_retries(
        issue_statement,
        test_str,
        patch_str,
        orig_stdout,
        orig_stderr,
        patched_stdout,
        patched_stderr,
        retries=3,
    )
    review, thread = next(gen)
    # Verify that a valid Review is returned.
    assert review is not None
    assert review.patch_decision == ReviewDecision.YES
    # Verify that the thread includes the SYSTEM_PROMPT.
    assert any(
        SYSTEM_PROMPT in msg["content"]
        for msg in thread.messages
        if msg["role"] == "system"
    )


@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
def test_run_with_retries_multiple_attempts():
    """
    Test run_with_retries when the model returns an invalid response on the first attempt,
    and a valid JSON review on the next attempt.
    """
    invalid_json = "invalid json"
    valid_json = json.dumps(
        {
            "patch-correct": "no",
            "patch-analysis": "patch is not good",
            "patch-advice": "fix patch",
            "test-correct": "yes",
            "test-analysis": "test works",
            "test-advice": "",
        }
    )
    responses = [invalid_json, valid_json]
    dummy_model = DummyModel(responses)
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL = dummy_model

    issue_statement = "Test issue"
    test_str = "dummy test"
    patch_str = "dummy patch"
    orig_stdout = "orig stdout"
    orig_stderr = "orig stderr"
    patched_stdout = "patched stdout"
    patched_stderr = "patched stderr"
    gen = run_with_retries(
        issue_statement,
        test_str,
        patch_str,
        orig_stdout,
        orig_stderr,
        patched_stdout,
        patched_stderr,
        retries=3,
    )
    # First yield should be invalid.
    review, thread = next(gen)
    assert review is None
    # Next yield should be valid.
    review, thread = next(gen)
    assert review is not None
    assert review.patch_decision == ReviewDecision.NO
    assert review.patch_advice == "fix patch"


###############################################################################
# Tests for run
###############################################################################
@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
def test_run_successful():
    """
    Test that run returns a valid review and thread when run_with_retries yields a valid review.
    """
    valid_json = json.dumps(
        {
            "patch-correct": "yes",
            "patch-analysis": "patch is correct",
            "patch-advice": "",
            "test-correct": "yes",
            "test-analysis": "test is valid",
            "test-advice": "",
        }
    )
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL.call.return_value = (valid_json,)

    issue_statement = "Test issue"
    test_content = "dummy test"
    patch_content = "dummy patch"
    dummy_repro = ReproResult("stdout", "stderr", 0)
    dummy_repro.reproduced = True
    review, thread = run(
        issue_statement, test_content, patch_content, dummy_repro, dummy_repro
    )
    assert review is not None
    assert review.patch_decision == ReviewDecision.YES


@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
def test_run_failure():
    """
    Test that run raises an InvalidLLMResponse when run_with_retries never yields a valid review.
    """
    invalid_response = "invalid json"
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL.call.return_value = (invalid_response,)

    issue_statement = "Test issue"
    test_content = "dummy test"
    patch_content = "dummy patch"
    dummy_repro = ReproResult("stdout", "stderr", 0)
    dummy_repro.reproduced = True
    with pytest.raises(InvalidLLMResponse):
        run(issue_statement, test_content, patch_content, dummy_repro, dummy_repro)
