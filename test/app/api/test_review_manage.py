import json
import tempfile
import textwrap
from pathlib import Path
from collections.abc import Generator

import pytest
from loguru import logger

from app.agents.agent_common import InvalidLLMResponse
from app.agents.agent_reviewer import Review, ReviewDecision
from app.agents.agent_write_patch import PatchAgent, PatchHandle
from app.agents.agent_reproducer import TestAgent, TestHandle
from app.data_structures import BugLocation, MessageThread, ReproResult
from app.api import review_manage
from app.api.review_manage import ReviewManager
from app.search.search_manage import SearchManager
from app.task import SweTask, Task

from test.pytest_utils import DummyTask

# --- Dummy Objects for Testing ---


class DummyPatchHandle(PatchHandle):
    def __str__(self):
        return "dummy_patch"


class DummyTestHandle(TestHandle):
    def __str__(self):
        return "dummy_test"


class DummyReproResult(ReproResult):
    def __init__(self):
        self.stdout = "out"
        self.stderr = "err"
        self.returncode = 0
        self.reproduced = True


class DummyTestAgent(TestAgent):
    def __init__(self):
        # Simulate no history initially.
        self._history = []
        self._tests = {}
        # Record feedback calls for later inspection.
        self.feedback = {}

    def write_reproducing_test_without_feedback(self):
        # Return a test handle, content, and a dummy reproduction result.
        test_handle = DummyTestHandle()
        test_content = "def test_dummy(): pass"
        self._history.append(test_handle)
        self._tests[test_handle] = test_content
        return test_handle, test_content, DummyReproResult()

    def save_test(self, test_handle: TestHandle) -> None:
        # For testing, we simply record that save_test was called.
        self._tests[test_handle] = self._tests.get(test_handle, "saved")

    def add_feedback(self, test_handle: TestHandle, feedback: str) -> None:
        self.feedback[str(test_handle)] = feedback


class DummyPatchAgent(PatchAgent):
    def __init__(
        self, task, search_manager, issue_stmt, context_thread, bug_locs, output_dir
    ):
        # For testing, we record calls.
        self.task = task
        self.search_manager = search_manager
        self.issue_stmt = issue_stmt
        self.context_thread = context_thread
        self.bug_locs = bug_locs
        self.output_dir = output_dir
        self.feedback = {}

    def write_applicable_patch_without_feedback(self):
        # Return a dummy patch handle and content.
        return DummyPatchHandle(), "patch content v1"

    def write_applicable_patch_with_feedback(self):
        # Return a new dummy patch after feedback.
        return DummyPatchHandle(), "patch content v2"

    def add_feedback(self, patch_handle: PatchHandle, feedback: str):
        self.feedback[str(patch_handle)] = feedback


class DummyReview(Review):
    def __init__(self, patch_decision, test_decision):
        self.patch_decision = patch_decision
        self.test_decision = test_decision
        self.patch_analysis = "analysis"
        self.patch_advice = "advice"
        self.test_analysis = "test analysis"
        self.test_advice = "test advice"

    def to_json(self):
        return {
            "patch_decision": self.patch_decision.name,
            "test_decision": self.test_decision.name,
            "patch_analysis": self.patch_analysis,
            "patch_advice": self.patch_advice,
            "test_analysis": self.test_analysis,
            "test_advice": self.test_advice,
        }


class DummyReviewThread:
    def __init__(self):
        self.messages = []

    def save_to_file(self, path: Path) -> None:
        # Write a dummy JSON content.
        path.write_text(json.dumps({"dummy": "review_thread"}, indent=4))


# DummySweTask for testing the generator.
class DummySweTask(SweTask):
    def __init__(self, issue_statement: str, output_dir: str):
        self._issue_statement = issue_statement
        self.output_dir = output_dir

    def get_issue_statement(self) -> str:
        return self._issue_statement

    def execute_reproducer(self, test_content: str, patch_content: str) -> ReproResult:
        # Return a dummy reproduction result that changes with patch content.
        result = DummyReproResult()
        if "v2" in patch_content:
            result.stdout = "changed out"
        return result


# Dummy SearchManager: fix by accepting project_path and output_dir.
class DummySearchManager(SearchManager):
    def __init__(self, project_path="dummy_project", output_dir="dummy_output"):
        self.project_path = project_path
        self.output_dir = output_dir


# Dummy MessageThread.
class DummyMessageThread(MessageThread):
    def __init__(self):
        self.messages = []

    def add_system(self, content: str):
        pass

    def add_user(self, content: str):
        pass

    def add_model(self, content: str, attachments: list):
        pass


# Monkey-patch print_review and print_acr to do nothing during tests.
def dummy_print_review(msg: str):
    pass


def dummy_print_acr(msg: str, title: str):
    pass


@pytest.fixture(autouse=True)
def monkey_patch_prints(monkeypatch):
    monkeypatch.setattr(review_manage, "print_review", dummy_print_review)
    monkeypatch.setattr(review_manage, "print_acr", dummy_print_acr)


# Monkey-patch agent_reviewer.run to return our dummy review and review thread.
@pytest.fixture(autouse=True)
def monkey_patch_reviewer(monkeypatch):
    def fake_run(
        issue_stmt, test_content, patch_content, orig_repro_result, patched_repro_result
    ):
        # For testing, we return a review with YES decisions.
        review = DummyReview(ReviewDecision.YES, ReviewDecision.YES)
        review_thread = DummyReviewThread()
        return review, review_thread

    monkeypatch.setattr(review_manage.agent_reviewer, "run", fake_run)


# --- Test Cases ---


def test_compose_feedback_for_patch_generation():
    dummy_review = DummyReview(ReviewDecision.NO, ReviewDecision.YES)
    test_content = "def test_example(): pass"
    feedback = ReviewManager.compose_feedback_for_patch_generation(
        dummy_review, test_content
    )
    # Check that the feedback message contains the test content and parts from the review.
    assert "test_example" in feedback
    assert "analysis" in feedback
    assert "advice" in feedback


def test_compose_feedback_for_test_generation():
    dummy_review = DummyReview(ReviewDecision.YES, ReviewDecision.NO)
    patch = "patch content"
    feedback = ReviewManager.compose_feedback_for_test_generation(dummy_review, patch)
    assert "patch content" in feedback
    assert "test analysis" in feedback
    assert "test advice" in feedback


def test_patch_only_generator(tmp_path):
    # Set up a temporary output directory.
    output_dir = str(tmp_path / "output")
    Path(output_dir).mkdir()

    # Create a dummy MessageThread.
    thread = DummyMessageThread()
    # Create a dummy bug location list.
    bug_locs = []

    # Create a dummy SearchManager with required args.
    search_manager = DummySearchManager(
        project_path="dummy_project", output_dir=output_dir
    )

    # Create a dummy Task (non-SweTask is fine for patch_only_generator).
    # This uses the DummyTask class from pytest_utils.py.
    task = DummyTask()

    # Create a dummy test agent.
    test_agent = DummyTestAgent()

    # Create a dummy patch agent.
    dummy_patch_agent = DummyPatchAgent(
        task, search_manager, task.get_issue_statement(), thread, bug_locs, output_dir
    )

    # Create our ReviewManager and override its patch_agent with our dummy.
    manager = ReviewManager(
        thread, bug_locs, search_manager, task, output_dir, test_agent
    )
    manager.patch_agent = dummy_patch_agent

    gen = manager.patch_only_generator()
    # Get first yielded patch.
    patch_handle, patch_content = next(gen)
    # Verify that the saved patch file exists.
    patch_file = Path(output_dir, f"extracted_patch_{patch_handle}.diff")
    assert patch_file.read_text() == patch_content

    # Now, force the generator to abort by making write_applicable_patch_without_feedback raise an exception.
    def raise_invalid():
        raise InvalidLLMResponse("dummy error")

    dummy_patch_agent.write_applicable_patch_without_feedback = raise_invalid
    # Advance generator to trigger exception and termination.
    with pytest.raises(StopIteration):
        next(gen)


def test_generator_with_patch_decision_yes(tmp_path):
    # Set up a temporary output directory.
    output_dir = str(tmp_path / "output")
    Path(output_dir).mkdir()

    # Create a dummy MessageThread.
    thread = DummyMessageThread()
    bug_locs = []
    search_manager = DummySearchManager(
        project_path="dummy_project", output_dir=output_dir
    )
    # Create a dummy SweTask.
    task = DummySweTask("dummy issue statement", output_dir)

    # Create dummy test agent and patch agent.
    test_agent = DummyTestAgent()
    dummy_patch_agent = DummyPatchAgent(
        task, search_manager, task.get_issue_statement(), thread, bug_locs, output_dir
    )

    manager = ReviewManager(
        thread, bug_locs, search_manager, task, output_dir, test_agent
    )
    manager.patch_agent = dummy_patch_agent

    # Prepare a generator from the full generator.
    gen: Generator = manager.generator(rounds=1)

    # The generator should first write the test if history is empty.
    # Then, it writes the first patch and then goes into the loop.
    # Advance generator until we reach a yield for a patch decision YES.
    yielded = next(gen)
    # At this point, yielded is from the first iteration of _generator.
    # Now, simulate sending an evaluation message.
    evaluation_msg = "Evaluation OK"
    try:
        next_val = gen.send(evaluation_msg)
    except StopIteration:
        next_val = None

    # Check that the evaluation message was used to add feedback.
    # Our dummy_patch_agent.feedback should have a key corresponding to the patch_handle.
    assert "dummy_patch" in dummy_patch_agent.feedback
    # Also, check that a review file has been written.
    review_file = Path(output_dir, f"review_pdummy_patch_tdummy_test.json")
    assert review_file.is_file()

    # Also, check that an execution result file has been written.
    exec_file = Path(output_dir, f"execution_dummy_patch_dummy_test.json")
    assert exec_file.is_file()
