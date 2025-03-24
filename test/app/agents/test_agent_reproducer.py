import json
import re
from pathlib import Path
import tempfile

import pytest
from unittest.mock import MagicMock, patch

# Import the items to be tested.
from app.agents.agent_reproducer import (
    SYSTEM_PROMPT,
    INITIAL_REQUEST,
    NoReproductionStep,
    TestAgent,
    generator,
    extract_markdown_code_blocks,
)
from app.data_structures import MessageThread, ReproResult
from app.task import Task
from test.pytest_utils import *

###############################################################################
# Tests that require SELECTED_MODEL are patched via patch.dict on app.model.common.__dict__
###############################################################################


###############################################################################
# Test 1: _issue_has_reproduction_steps returns True when a reproducible example exists.
###############################################################################
@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
def test_issue_has_reproduction_steps_true():
    """
    Simulate a model response indicating that the issue contains a reproducible example.
    Verify that the method returns True and that the constructed MessageThread contains
    the expected system and user messages.
    """
    # Set up the dummy response.
    dummy_response = '{"has-reproducible-example": true}'
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL.call.return_value = (dummy_response,)

    issue_statement = "Issue with reproducible steps."
    reproducible, thread = TestAgent._issue_has_reproduction_steps(issue_statement)
    assert reproducible is True
    # Verify that the thread contains both system and user messages.
    contents = [
        msg["content"] for msg in thread.messages if msg["role"] in ("system", "user")
    ]
    assert any("You are an experienced software engineer" in c for c in contents)
    assert any("Here is an issue:" in c for c in contents)


###############################################################################
# Test 2: _issue_has_reproduction_steps returns False when no reproducible example exists.
###############################################################################
@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
def test_issue_has_reproduction_steps_false():
    """
    Simulate a model response indicating that the issue does not contain a reproducible example.
    Verify that the method returns False.
    """
    dummy_response = '{"has-reproducible-example": false}'
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL.call.return_value = (dummy_response,)

    issue_statement = "Issue without reproducible steps."
    reproducible, _ = TestAgent._issue_has_reproduction_steps(issue_statement)
    assert reproducible is False


###############################################################################
# Test 3: convert_response_to_test returns expected test content for various cases.
###############################################################################
def test_convert_response_to_test():
    """
    Test convert_response_to_test for:
      - A response with exactly one code block.
      - A response with two code blocks where the second is the reproducer command.
      - A response with two code blocks where the second is not the reproducer command.
      - A response with no code blocks.
    """
    # Single code block case.
    response_single = "```\nprint('hello')\n```"
    result = TestAgent.convert_response_to_test(response_single)
    assert result.strip() == "print('hello')"

    # Two code blocks; second is the reproducer command.
    response_two = "```\nprint('hello')\n```\n```\npython3 reproducer.py\n```"
    result = TestAgent.convert_response_to_test(response_two)
    assert result.strip() == "print('hello')"

    # Two code blocks; second is not the reproducer command.
    response_invalid = "```\nprint('hello')\n```\n```\nother command\n```"
    result = TestAgent.convert_response_to_test(response_invalid)
    assert result is None

    # No code blocks.
    response_none = "No code here"
    result = TestAgent.convert_response_to_test(response_none)
    assert result is None


###############################################################################
# Test 4: _write_test constructs the thread and extracts test content correctly.
###############################################################################
@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
@patch("app.agents.agent_reproducer.print_acr")
def test_write_test(mock_print_acr):
    """
    Verify that _write_test:
      - Constructs a MessageThread with proper prompts.
      - Calls SELECTED_MODEL.call to get a response.
      - Correctly extracts test content from the model response.
      - Calls print_acr when no history handles are provided.
    """
    dummy_response = "```\nprint('test')\n```"
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL.call.return_value = (dummy_response,)

    # Use the dummy Task imported from pytest_utils.
    dummy_task = DummyTask()
    with tempfile.TemporaryDirectory() as temp_dir:
        agent = TestAgent(dummy_task, temp_dir)
        response, test_content, thread = agent._write_test([])
        assert test_content.strip() == "print('test')"
        mock_print_acr.assert_called_with(INITIAL_REQUEST)


###############################################################################
# Test 5: Generator function for reproducer test generation.
###############################################################################
@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
@patch("app.agents.agent_reproducer.print_acr")
@patch("app.agents.agent_reproducer.print_reproducer")
def test_generator_function(mock_print_reproducer, mock_print_acr):
    """
    Simulate two iterations of the generator:
      - First iteration: model returns a response with two code blocks (invalid reproduction),
        so the generator yields an empty test content with a False flag.
      - Second iteration: model returns a response with exactly one code block,
        so the generator yields valid test content with a True flag.
    """
    response_two_blocks = "```\nblock1\n```\n```\nblock2\n```"
    response_one_block = "```\ncorrect\n```"
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL.call.side_effect = [
        (response_two_blocks,),
        (response_one_block,),
        ("extra",),  # extra response to cover any further call.
    ]
    issue_statement = "Test issue"
    gen = generator(issue_statement)

    # First iteration: yields an empty test (invalid reproduction).
    test_content, thread, flag = next(gen)
    assert test_content == ""
    assert flag is False

    # Send a feedback message to continue the loop.
    evaluation_msg = "error details"
    test_content2, thread2, flag2 = gen.send(evaluation_msg)
    # Second iteration: valid test content.
    assert test_content2.strip() == "correct"
    assert flag2 is True

    try:
        gen.close()
    except Exception:
        pass


###############################################################################
# Test 6: _select_feedback_handles returns correct feedback handles.
###############################################################################
def test_select_feedback_handles():
    """
    Test _select_feedback_handles with different requested numbers.
    """
    agent = TestAgent(task=MagicMock(), task_dir="dummy")
    agent._history = ["h1", "h2"]
    agent._non_repro_history = ["n1"]
    # Requesting 1 feedback: last handle from _history.
    handles = agent._select_feedback_handles(1)
    assert handles == ["h2"]
    # Requesting 3 feedbacks: non-repro history followed by _history.
    handles = agent._select_feedback_handles(3)
    assert handles == ["n1", "h1", "h2"]
    # Requesting more than available: all handles.
    handles = agent._select_feedback_handles(10)
    assert handles == ["n1", "h1", "h2"]


###############################################################################
# Test 7: _register_reproducing_test registers the test correctly.
###############################################################################
def test_register_reproducing_test():
    """
    Verify that _register_reproducing_test:
      - Stores the response and test content.
      - Adds the handle to _history.
    """
    agent = TestAgent(task=MagicMock(), task_dir="dummy")
    agent._request_idx = 0
    response = "response"
    test_content = "print('hello')"
    handle = agent._register_reproducing_test(response, test_content)
    assert handle == "0"
    assert agent._responses[handle] == response
    assert agent._tests[handle] == test_content
    assert handle in agent._history


###############################################################################
# Test 8: _register_non_reproducing_test registers a failing test and adds feedback.
###############################################################################
def test_register_non_reproducing_test():
    """
    Verify that _register_non_reproducing_test:
      - Stores the response and test content.
      - Adds the handle to _non_repro_history.
      - Appends appropriate feedback from the reproduction result.
    """
    agent = TestAgent(task=MagicMock(), task_dir="dummy")
    agent._request_idx = 1
    response = "response"
    test_content = "print('fail')"
    dummy_repro = ReproResult("out", "err", 1)
    dummy_repro.reproduced = False
    handle = agent._register_non_reproducing_test(response, test_content, dummy_repro)
    assert handle == "1"
    assert agent._responses[handle] == response
    assert agent._tests[handle] == test_content
    assert handle in agent._non_repro_history
    feedback = agent._feedbacks[handle][0]
    assert "did not reproduce" in feedback
    assert "1" in feedback


###############################################################################
# Test 9: add_feedback raises ValueError for an unknown test handle.
###############################################################################
def test_add_feedback_raises():
    """
    Verify that add_feedback raises a ValueError when adding feedback for a non-existent handle.
    """
    agent = TestAgent(task=MagicMock(), task_dir="dummy")
    with pytest.raises(ValueError):
        agent.add_feedback("nonexistent", "some feedback")


###############################################################################
# Test 10: _construct_init_thread returns a thread with proper initial messages.
###############################################################################
def test_construct_init_thread():
    """
    Verify that _construct_init_thread creates a MessageThread containing:
      - The system prompt.
      - A user message with the issue statement.
    """
    agent = TestAgent(task=MagicMock(), task_dir="dummy")
    thread = agent._construct_init_thread()
    roles = [msg["role"] for msg in thread.messages]
    contents = [msg["content"] for msg in thread.messages]
    assert roles[0] == "system"
    assert "You are an experienced software engineer" in contents[0]
    assert any("Here is an issue:" in content for content in contents)


###############################################################################
# Test 11: _feedback_from_repro_result returns the expected formatted string.
###############################################################################
def test_feedback_from_repro_result():
    """
    Verify that _feedback_from_repro_result returns a string that includes
    the return code, stdout, and stderr from the reproduction result.
    """
    agent = TestAgent(task=MagicMock(), task_dir="dummy")
    dummy_repro = ReproResult("stdout content", "stderr content", 2)
    dummy_repro.reproduced = False
    feedback = agent._feedback_from_repro_result(dummy_repro)
    assert "2" in feedback
    assert "stdout content" in feedback
    assert "stderr content" in feedback


###############################################################################
# Test 12: save_test writes the test file correctly.
###############################################################################
def test_save_test(tmp_path):
    """
    Verify that save_test writes the test content to a file in the task directory.
    Uses pytest's tmp_path fixture.
    """
    agent = TestAgent(task=MagicMock(), task_dir=str(tmp_path))
    agent._request_idx = 0
    response = "response"
    test_content = "print('saved test')"
    handle = agent._register_reproducing_test(response, test_content)
    agent.save_test(handle)
    file_path = Path(tmp_path, f"reproducer_{handle}.py")
    assert file_path.exists()
    content = file_path.read_text()
    assert "print('saved test')" in content
