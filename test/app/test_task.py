import contextlib
import json
import os
import subprocess
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from tempfile import TemporaryDirectory

import pytest

from app.task import Task, SweTask, PlainTask
import app.utils as apputils
import app.config as config
from app.data_structures import ReproResult
from test.pytest_utils import *

from test.pytest_utils import DummyTask as BaseDummyTask


# Extend the existing DummyTask to accept extra arguments without breaking behavior.
# This new class is “open for extension” per the Open/Closed Principle.
class DummyTask(BaseDummyTask):
    def __init__(
        self, project_path="dummy_project", issue="dummy issue", env_name="dummy_env"
    ):
        super().__init__(project_path, issue)
        self.env_name = env_name


# -----------------------------------------------------------------------------
# Helper: Fake cd context manager that does nothing
# -----------------------------------------------------------------------------
@contextlib.contextmanager
def fake_cd(newdir):
    yield


# -----------------------------------------------------------------------------
# Tests for PlainTask
# -----------------------------------------------------------------------------
def test_plain_task_getters_and_reset(monkeypatch, tmp_path):
    # Use a temporary directory as the project directory.
    local_dir = tmp_path / "plain_project"
    local_dir.mkdir()
    commit = "dummy_commit"
    problem = "dummy problem"
    task = PlainTask(
        commit_hash=commit, local_path=str(local_dir), problem_statement=problem
    )

    # Test getters.
    assert task.get_issue_statement() == problem
    assert task.project_path == str(local_dir)

    # Patch cd to be a no-op.
    monkeypatch.setattr(apputils, "cd", fake_cd)

    # Patch repo_reset_and_clean_checkout to record calls.
    calls = []

    def fake_reset(commit_arg):
        calls.append(commit_arg)

    monkeypatch.setattr(apputils, "repo_reset_and_clean_checkout", fake_reset)

    task.setup_project()
    assert calls == [commit]

    calls.clear()
    task.reset_project()
    assert calls == [commit]

    # Validate is not implemented.
    with pytest.raises(NotImplementedError):
        task.validate("dummy patch")


# -----------------------------------------------------------------------------
# Fixture: Create a dummy SweTask instance
# -----------------------------------------------------------------------------
@pytest.fixture
def dummy_swe_task(monkeypatch, tmp_path):
    # Create a temporary directory to simulate a repository.
    repo_dir = tmp_path / "swe_repo"
    repo_dir.mkdir()
    # Create a dummy .gitignore (required for make_noop_patch)
    (repo_dir / ".gitignore").write_text("ignored.txt")

    params = {
        "task_id": "dummy_task",
        "problem_statement": "dummy problem",
        "repo_path": str(repo_dir),
        "commit": "dummy_commit",
        "env_name": "dummy_env",
        "repo_name": "dummy_repo",
        "repo_version": "v1.0",
        "pre_install_cmds": [],
        "install_cmd": "",
        "test_cmd": "echo test",
        "test_patch": "dummy_patch",
        "testcases_passing": [],
        "testcases_failing": [],
    }
    # Set a config flag so that _do_install is executed.
    config.enable_sbfl = True

    # Patch cd and other git functions to avoid real commands.
    monkeypatch.setattr(apputils, "cd", fake_cd)
    monkeypatch.setattr(apputils, "repo_reset_and_clean_checkout", lambda commit: None)
    monkeypatch.setattr(apputils, "repo_commit_current_changes", lambda: None)

    # Patch run_string_cmd_in_conda to simulate successful install commands.
    def fake_run_string_cmd_in_conda(cmd, env, **kwargs):
        return CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(
        apputils, "run_string_cmd_in_conda", fake_run_string_cmd_in_conda
    )

    return SweTask(**params)


def test_swe_task_getters_and_setup(dummy_swe_task, monkeypatch):
    # Test getters.
    assert dummy_swe_task.get_issue_statement() == "dummy problem"
    assert dummy_swe_task.project_path == dummy_swe_task.repo_path

    # Patch _do_install to record its call.
    install_called = False

    def fake_install():
        nonlocal install_called
        install_called = True

    monkeypatch.setattr(dummy_swe_task, "_do_install", fake_install)

    dummy_swe_task.setup_project()
    assert install_called is True

    # Test reset_project by recording calls.
    calls = []
    monkeypatch.setattr(
        apputils, "repo_reset_and_clean_checkout", lambda commit: calls.append(commit)
    )
    dummy_swe_task.reset_project()
    assert calls == [dummy_swe_task.commit]


def test_swe_task_validate(monkeypatch, dummy_swe_task, tmp_path):
    # Test validate method by patching inner functions.
    # We'll have our fake _run_test_suite_for_regression_docker accept both patch_content and log_file.
    def fake_run_test_suite_docker(patch_content, log_file):
        # Return a tuple with (tests_passed, message, orig_log_file)
        return (True, "dummy message", "dummy_orig_log_file")

    monkeypatch.setattr(
        dummy_swe_task,
        "_run_test_suite_for_regression_docker",
        fake_run_test_suite_docker,
    )

    # Patch apply_patch to be a no-op by returning a nullcontext.
    from contextlib import nullcontext

    monkeypatch.setattr(dummy_swe_task, "apply_patch", lambda patch: nullcontext())

    # Call validate; note that validate internally calls mkstemp to create a log file.
    tests_passed, msg, log_file, orig_log_file = dummy_swe_task.validate("patch")
    assert tests_passed is True
    assert msg == "dummy message"
    # We expect orig_log_file to be our dummy value.
    assert orig_log_file == "dummy_orig_log_file"
    # log_file should be a string (its value comes from mkstemp, so we can't predict it exactly).
    assert isinstance(log_file, str)


def test_swe_task_make_noop_patch(monkeypatch, tmp_path):
    # Test the make_noop_patch class method.
    d = tmp_path / "dummy_repo"
    d.mkdir()
    # Create a dummy .gitignore file.
    gitignore = d / ".gitignore"
    gitignore.write_text("ignored_file.txt")

    # Patch subprocess.run to simulate git commands.
    def fake_run(cmd, cwd, **kwargs):
        if cmd[:3] == ["git", "diff", "HEAD~"]:
            return CompletedProcess(cmd, 0, stdout="noop diff", stderr="")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    noop_patch = SweTask.make_noop_patch(str(d))
    assert "noop diff" in noop_patch


# -----------------------------------------------------------------------------
# Test for Task.apply_patch context manager.
# -----------------------------------------------------------------------------
def test_apply_patch(monkeypatch, tmp_path):
    # Create a dummy subclass of Task to test apply_patch.
    class DummyApplyTask(Task):
        @property
        def project_path(self) -> str:
            return str(tmp_path / "dummy_project")

        def get_issue_statement(self) -> str:
            return "dummy issue"

        def setup_project(self) -> None:
            pass

        def reset_project(self) -> None:
            pass

        def validate(self, patch_content: str):
            return True, "", "", ""

    # Create a dummy project directory.
    proj_dir = Path(tmp_path / "dummy_project")
    proj_dir.mkdir()
    # Create a dummy file inside the project.
    dummy_file = proj_dir / "dummy.txt"
    dummy_file.write_text("original content")

    # Patch cd to be a no-op.
    monkeypatch.setattr(apputils, "cd", fake_cd)

    # Patch subprocess.run used in apply_patch to simulate a successful git apply.
    def fake_run(cmd, **kwargs):
        return CompletedProcess(cmd, 0, stdout="applied", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Patch repo_clean_changes to record its call.
    clean_called = False

    def fake_clean():
        nonlocal clean_called
        clean_called = True

    monkeypatch.setattr(apputils, "repo_clean_changes", fake_clean)

    dummy = DummyApplyTask()
    with dummy.apply_patch("patch content"):
        # Nothing to do inside the context.
        pass
    assert clean_called, "Expected repo_clean_changes to be called after apply_patch"


# Dummy values for required fields in SweTask.
DUMMY_TASK_PARAMS = {
    "task_id": "dummy_task",
    "problem_statement": "dummy problem",
    # We'll use a temporary directory as repo_path in each test.
    "repo_path": None,
    "commit": "dummy_commit",
    "env_name": "dummy_env",
    "repo_name": "dummy_repo",
    "repo_version": "v1.0",
    "pre_install_cmds": [],
    "install_cmd": "",
    "test_cmd": "echo test",  # Not used in execute_reproducer
    "test_patch": "dummy_patch",
    "testcases_passing": [],
    "testcases_failing": [],
}


# -----------------------------------------------------------------------------
# Test execute_reproducer - normal execution case.
# -----------------------------------------------------------------------------
def test_execute_reproducer_normal(monkeypatch, tmp_path):
    # Create a temporary directory to simulate a repository.
    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()
    # Write a dummy file to ensure the directory is not empty.
    (repo_dir / "dummy.txt").write_text("original content")

    # Update the dummy parameters with the temporary repo path.
    params = DUMMY_TASK_PARAMS.copy()
    params["repo_path"] = str(repo_dir)

    # Instantiate a SweTask.
    task = SweTask(**params)

    # Patch run_script_in_conda to simulate a successful process execution.
    def fake_run_script_in_conda(args, env_name, cwd, **kwargs):
        # args is expected to be a list with a temporary filename.
        return CompletedProcess(args, 0, stdout="dummy stdout", stderr="dummy stderr")

    # Patch in the module where execute_reproducer looks it up.
    monkeypatch.setattr("app.task.run_script_in_conda", fake_run_script_in_conda)

    # For this test, we do not provide any patch content, so apply_patch is not used.
    test_content = "print('hello world')"
    result = task.execute_reproducer(test_content, patch_content=None)

    # Assert that the returned ReproResult matches our fake CompletedProcess.
    assert result.stdout == "dummy stdout"
    assert result.stderr == "dummy stderr"
    assert result.returncode == 0


# -----------------------------------------------------------------------------
# Test execute_reproducer - timeout case.
# -----------------------------------------------------------------------------
def test_execute_reproducer_timeout(monkeypatch, tmp_path):
    # Create a temporary dummy repo directory.
    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()

    params = DUMMY_TASK_PARAMS.copy()
    params["repo_path"] = str(repo_dir)

    task = SweTask(**params)

    # Patch run_script_in_conda to simulate a timeout.
    def fake_run_script_in_conda(args, env_name, cwd, **kwargs):
        raise TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 120))

    monkeypatch.setattr("app.task.run_script_in_conda", fake_run_script_in_conda)

    test_content = "print('hello world')"
    result = task.execute_reproducer(test_content, patch_content=None)

    # In the timeout case, we expect:
    # - stdout to be empty,
    # - stderr to equal "Test execution timeout.",
    # - returncode to be -1.
    assert result.stdout == ""
    assert result.stderr == "Test execution timeout."
    assert result.returncode == -1
