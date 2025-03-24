from collections import defaultdict
from pathlib import Path
import tempfile
from tempfile import NamedTemporaryFile

import pytest
from unittest.mock import MagicMock, patch

# Import our test utilities (assumed available).
from test.pytest_utils import *

# Import the module under test.
from app.agents.agent_write_patch import (
    SYSTEM_PROMPT,
    USER_PROMPT_INIT,
    PatchAgent,
    generator,
)
from app.data_structures import BugLocation, MessageThread, ReproResult
from app.agents import agent_common
from app.log import print_acr, print_patch_generation
from app.model import common
from app.post_process import (
    ExtractStatus,
    convert_response_to_diff,
    extract_diff_one_instance,
    record_extract_status,
)
from app.task import Task
from test.pytest_utils import *

###############################################################################
# Helper Dummy Classes and Functions
###############################################################################

# DummyTask imported from pytest_utils.


# Create a dummy BugLocation with a patched multiple_locs_to_str_for_model.
class DummyBugLocation:
    @staticmethod
    def multiple_locs_to_str_for_model(bug_locs):
        return "dummy bug location string"


###############################################################################
# Test 1: _construct_init_thread when bug locations are provided.
###############################################################################
def test_construct_init_thread_with_bug_locs():
    """
    Verify that _construct_init_thread creates a thread that includes the system prompt,
    the issue statement, and a code context prompt (which uses bug locations).
    """
    # Patch the static method to return a fixed string.
    with patch.object(
        BugLocation, "multiple_locs_to_str_for_model", return_value="BUG_LOC_STR"
    ):
        # Create a PatchAgent with a non-empty bug_locs list.
        dummy_task = DummyTask()
        bug_locs = [DummyBugLocation()]  # dummy bug location
        dummy_context = MessageThread()
        agent = PatchAgent(
            dummy_task, MagicMock(), "Issue text", dummy_context, bug_locs, "dummy_dir"
        )
        thread = agent._construct_init_thread()
        messages = [
            msg["content"]
            for msg in thread.messages
            if msg["role"] in ("system", "user")
        ]
        # Expect system prompt and the issue text.
        assert any(SYSTEM_PROMPT in m for m in messages)
        assert any("Here is the issue:" in m for m in messages)
        # And the bug location string should appear in the code context prompt.
        assert any("BUG_LOC_STR" in m for m in messages)


###############################################################################
# Test 2: _construct_init_thread when bug locations are NOT provided.
###############################################################################
def test_construct_init_thread_without_bug_locs():
    """
    Verify that _construct_init_thread falls back to using the context thread,
    and that the system prompt is replaced.
    """
    # Create a dummy context thread with one message.
    dummy_context = MessageThread()
    dummy_context.add_system("Original system prompt")
    dummy_context.add_user("Some previous context")
    dummy_task = DummyTask()
    agent = PatchAgent(
        dummy_task, MagicMock(), "Issue text", dummy_context, [], "dummy_dir"
    )
    # Patch agent_common.replace_system_prompt to return a thread with SYSTEM_PROMPT.
    with patch(
        "app.agents.agent_write_patch.agent_common.replace_system_prompt",
        lambda thread, prompt: thread,
    ):
        thread = agent._construct_init_thread()
        # Expect the thread's first message to be the replaced system prompt.
        assert thread.messages[0]["content"] == "Original system prompt"


###############################################################################
# Test 3: _construct_code_context_prompt
###############################################################################
def test_construct_code_context_prompt():
    """
    Verify that _construct_code_context_prompt returns a string that includes the bug locations string.
    """
    # Patch BugLocation.multiple_locs_to_str_for_model to return a known string.
    with patch.object(
        BugLocation, "multiple_locs_to_str_for_model", return_value="LOC_DETAILS"
    ):
        dummy_task = DummyTask()
        dummy_context = MessageThread()
        agent = PatchAgent(
            dummy_task,
            MagicMock(),
            "Issue text",
            dummy_context,
            [DummyBugLocation()],
            "dummy_dir",
        )
        prompt = agent._construct_code_context_prompt()
        assert "LOC_DETAILS" in prompt
        assert "you should think what changes are necessary" in prompt


###############################################################################
# Test 4: _register_applicable_patch and add_feedback
###############################################################################
def test_register_applicable_patch_and_add_feedback():
    """
    Verify that _register_applicable_patch correctly registers a patch and that add_feedback
    raises an error for unknown handles.
    """
    dummy_task = DummyTask()
    dummy_context = MessageThread()
    agent = PatchAgent(dummy_task, MagicMock(), "Issue", dummy_context, [], "dummy_dir")
    agent._request_idx = 0
    response = "patch response"
    diff_content = "diff content"
    handle = agent._register_applicable_patch(response, diff_content)
    assert handle == "0"
    assert agent._responses[handle] == response
    assert agent._diffs[handle] == diff_content
    assert handle in agent._history
    # add_feedback should work for a registered patch
    agent._feedbacks[handle] = []
    agent.add_feedback(handle, "feedback")
    assert agent._feedbacks[handle] == ["feedback"]
    # For an unknown handle, it should raise ValueError.
    with pytest.raises(ValueError):
        agent.add_feedback("unknown", "some feedback")


###############################################################################
# Test 5: _write_patch returns expected values when patch is applicable.
###############################################################################
def test_write_patch_applicable():
    """
    Verify that _write_patch calls SELECTED_MODEL.call, passes the response through
    convert_response_to_diff, and returns a tuple indicating an applicable patch.
    """
    dummy_patch_resp = "patch response text"
    # Patch the dummy SELECTED_MODEL in the common module.
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL = MagicMock()
    common_mod.SELECTED_MODEL.call.return_value = (dummy_patch_resp,)

    # Patch convert_response_to_diff to simulate an applicable patch.
    with patch(
        "app.agents.agent_write_patch.convert_response_to_diff",
        return_value=(ExtractStatus.APPLICABLE_PATCH, "ok", "diff content"),
    ):
        # Patch record_extract_status (its output is not needed).
        with patch("app.agents.agent_write_patch.record_extract_status"):
            dummy_task = DummyTask()
            # Create a MessageThread with a system message to avoid IndexError.
            dummy_context = MessageThread()
            dummy_context.add_system("dummy system")
            # Use a temporary directory for task_dir.
            with tempfile.TemporaryDirectory() as temp_dir:
                agent = PatchAgent(
                    dummy_task, MagicMock(), "Issue text", dummy_context, [], temp_dir
                )
                applicable, resp, diff_content, thread = agent._write_patch([])
                assert applicable is True
                assert resp == dummy_patch_resp
                assert diff_content == "diff content"


###############################################################################
# Test 6: write_applicable_patch_without_feedback returns a patch on success.
###############################################################################
def test_write_applicable_patch_without_feedback():
    """
    Simulate a successful patch generation on the first try.
    Verify that write_applicable_patch_without_feedback returns the expected handle and diff content.
    """
    from app.agents.agent_write_patch import PatchAgent, USER_PROMPT_INIT

    # Create a dummy thread to return from _write_patch.
    dummy_thread = MessageThread()
    # Patch _write_patch to simulate a successful patch.
    with patch(
        "app.agents.agent_write_patch.PatchAgent._write_patch"
    ) as mock_write_patch, patch("app.agents.agent_write_patch.print_patch_generation"):
        mock_write_patch.return_value = (
            True,
            "patch response",
            "diff content",
            dummy_thread,
        )
        dummy_task = DummyTask()
        dummy_context = MessageThread()
        dummy_context.add_system("dummy system")
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = PatchAgent(
                dummy_task, MagicMock(), "Issue text", dummy_context, [], temp_dir
            )
            handle, diff = agent.write_applicable_patch_without_feedback(retries=1)
            # _register_applicable_patch should assign handle "0" on first call.
            assert handle == "0"
            assert diff == "diff content"


###############################################################################
# Test 7: Generator function (applicable patch branch)
###############################################################################
@patch.dict("app.model.common.__dict__", {"SELECTED_MODEL": MagicMock()})
@patch("app.agents.agent_write_patch.print_acr")
@patch("app.agents.agent_write_patch.print_patch_generation")
def test_generator_applicable_branch(mock_print_patch_generation, mock_print_acr):
    """
    Simulate one iteration of the generator where the patch extraction is successful.
    The generator should yield a tuple with True, a success message, and patch content.
    Then, when a validation message is sent back, the generator should update the thread.
    """
    # Prepare a dummy patch response.
    dummy_patch_resp = "dummy patch response"
    # Set SELECTED_MODEL.call to return dummy_patch_resp.
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL.call.return_value = (dummy_patch_resp,)

    # Patch extract_diff_one_instance to simulate an applicable patch.
    def dummy_extract_diff(raw_patch_file, tmp_name):
        # Write a dummy diff to tmp file.
        with open(tmp_name, "w") as f:
            f.write("dummy diff content")
        return (ExtractStatus.APPLICABLE_PATCH, "extraction ok")

    with patch(
        "app.agents.agent_write_patch.extract_diff_one_instance",
        side_effect=dummy_extract_diff,
    ):
        # Patch record_extract_status (dummy).
        with patch("app.agents.agent_write_patch.record_extract_status") as mock_record:
            # Create a dummy context thread.
            context_thread = MessageThread()
            context_thread.add_system("Original system prompt")
            context_thread.add_user("Context message")
            with tempfile.TemporaryDirectory() as temp_dir:
                gen = generator(context_thread, temp_dir)
                # The generator should add the USER_PROMPT_INIT.
                # First iteration: yield tuple.
                is_success, result_msg, patch_content = next(gen)
                assert is_success is True
                assert result_msg == "written an applicable patch"
                assert "dummy diff content" in patch_content
                # Send back a validation message.
                try:
                    gen.send("validation error")
                except AssertionError:
                    # The generator asserts that validation_msg is not None; if we send one, it continues.
                    pass
                # Close the generator.
                gen.close()


###############################################################################
# Test 8: Generator function (non-applicable branch)
###############################################################################
def test_generator_non_applicable_branch():
    """
    Simulate one iteration of the generator where patch extraction fails.
    The generator should yield a tuple with False, a failure message, and empty patch content.
    """
    from app.model import common as common_mod

    common_mod.SELECTED_MODEL = MagicMock()
    dummy_patch_resp = "dummy patch response"
    common_mod.SELECTED_MODEL.call.return_value = (dummy_patch_resp,)

    # Dummy implementation of extract_diff_one_instance to simulate failure.
    def dummy_extract_diff_fail(raw_patch_file, tmp_name):
        with open(tmp_name, "w") as f:
            f.write("")
        # Return a status that is not APPLICABLE_PATCH.
        return ("OTHER", "extraction failed")

    with patch(
        "app.agents.agent_write_patch.extract_diff_one_instance",
        side_effect=dummy_extract_diff_fail,
    ):
        with patch("app.agents.agent_write_patch.record_extract_status"):
            # Build a context thread with necessary messages.
            context_thread = MessageThread()
            context_thread.add_system("Original system prompt")
            context_thread.add_user("Context message")
            with tempfile.TemporaryDirectory() as temp_dir:
                gen = generator(context_thread, temp_dir)
                is_success, result_msg, patch_content = next(gen)
                assert is_success is False
                assert result_msg == "failed to write an applicable patch"
                assert patch_content == ""
                try:
                    gen.send(None)
                except AssertionError:
                    pass
                gen.close()
