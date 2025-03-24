import json
from pathlib import Path
from app.inference import *
from app.task import Task
from app import config

from test.pytest_utils import *  # Import shared test utilities


###############################################################################
# Define a DummyTask for testing purposes.
###############################################################################
class DummyTask(Task):
    def __init__(self, project_path, issue="dummy issue"):
        self._project_path = project_path
        self._issue = issue

    def get_issue_statement(self):
        return self._issue

    def reset_project(self):
        pass

    def setup_project(self):
        pass

    def validate(self):
        pass

    @property
    def project_path(self):
        return self._project_path


###############################################################################
# Dummy Review Manager for write_patch_iterative_with_review and patch_only_generator.
###############################################################################
class DummyReviewManager:
    # For review-based patch generation: yield two-tuples.
    def generator(self):
        # First attempt fails; second attempt succeeds.
        # (The caller uses the evaluation result from evaluate_patch to decide success.)
        yield ("patch1", "content1")
        yield ("patch2", "content2")

    # For patch-only generation: yield a two-tuple.
    def patch_only_generator(self):
        yield ("patchA", "contentA")


###############################################################################
# Test write_patch_iterative_with_review
###############################################################################
def test_write_patch_iterative_with_review(monkeypatch, tmp_path):
    """
    Simulate a successful patch generation with reviewer feedback.
    The dummy generator yields one failed attempt then one successful.
    Monkeypatch evaluation to return (False, ...) for the first patch and (True, ...) for the second.
    """

    # Define a dummy review manager that yields two-tuples.
    class DummyReviewManager:
        def generator(self):
            yield ("patch1", "content1")
            yield ("patch2", "content2")

    dummy_review_manager = DummyReviewManager()

    # Override evaluation: patch "patch1" fails, "patch2" passes.
    monkeypatch.setattr(
        "app.api.validation.evaluate_patch",
        lambda task, ph, pc, od: (
            (False, "fail") if ph == "patch1" else (True, "pass")
        ),
    )

    # Create dummy task and output directory.
    dummy_task = DummyTask(
        project_path=str(tmp_path / "dummy_project"), issue="dummy issue"
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = write_patch_iterative_with_review(
        dummy_task, str(output_dir), dummy_review_manager, retries=3
    )
    assert result is True
    # We no longer check for a file, since write_patch_iterative_with_review does not write agent_reproducer.json.


###############################################################################
# Test write_patch_iterative (without reviewer)
###############################################################################
def test_write_patch_iterative(monkeypatch, tmp_path):
    """
    Simulate a successful patch generation without reviewer.
    The dummy patch_only_generator yields one patch that passes evaluation.
    """
    dummy_review_manager = DummyReviewManager()
    monkeypatch.setattr(
        "app.api.validation.evaluate_patch", lambda task, ph, pc, od: (True, "pass")
    )
    dummy_task = DummyTask(
        project_path=str(tmp_path / "dummy_project"), issue="dummy issue"
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = write_patch_iterative(
        dummy_task, str(output_dir), dummy_review_manager, retries=3
    )
    assert result is True


###############################################################################
# Test run_one_task
###############################################################################
def test_run_one_task(monkeypatch, tmp_path):
    """
    Simulate overall workflow by:
      - Setting config.overall_retry_limit,
      - Overriding set_model to do nothing,
      - Overriding _run_one_task to always return True,
      - Overriding select_patch to return dummy values.
    Verify that run_one_task returns True and writes selected_patch.json.
    """
    # Set overall_retry_limit.
    monkeypatch.setattr(config, "overall_retry_limit", 2)
    # Override set_model so it does nothing.
    monkeypatch.setattr("app.inference.set_model", lambda model_name: None)
    # Override _run_one_task to always return True.
    monkeypatch.setattr(
        "app.inference._run_one_task", lambda out_dir, api_manager, issue_stmt: True
    )
    # Override select_patch to return dummy values.
    monkeypatch.setattr(
        "app.inference.select_patch",
        lambda task, output_dir: ("patch_selected", {"reason": "dummy reason"}),
    )

    dummy_task = DummyTask(
        project_path=str(tmp_path / "dummy_project"), issue="dummy issue"
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result = run_one_task(dummy_task, str(output_dir), ["dummy_model"])
    # Check that selected_patch.json is written.
    selected_patch_file = Path(output_dir) / "selected_patch.json"
    assert selected_patch_file.exists()
    data = json.loads(selected_patch_file.read_text())
    assert data["reason"] == "dummy reason"
    assert result is True


###############################################################################
# Test select_patch
###############################################################################
def test_select_patch(monkeypatch, tmp_path):
    """
    Create a temporary directory with a dummy extracted patch file and its corresponding review file.
    Monkeypatch may_pass_regression_tests to always return True.
    Verify that select_patch returns a tuple with a relative patch path and details that contain 'reviewer-approved'.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # Create a subdirectory to hold patch files.
    patch_dir = output_dir / "patches"
    patch_dir.mkdir()
    patch_file = patch_dir / "extracted_patch_1.diff"
    patch_file.write_text("dummy patch content")
    # Create a corresponding review file.
    review_file = patch_dir / "review_p1_t.json"
    review_file.write_text(json.dumps({"patch-correct": "yes"}))

    # Monkeypatch may_pass_regression_tests to always return True.
    monkeypatch.setattr("app.inference.may_pass_regression_tests", lambda task, p: True)

    dummy_task = DummyTask(
        project_path=str(tmp_path / "dummy_project"), issue="dummy issue"
    )
    selected_patch, details = select_patch(dummy_task, str(output_dir))
    assert "reviewer-approved" in details["reason"]
    assert isinstance(selected_patch, str)
