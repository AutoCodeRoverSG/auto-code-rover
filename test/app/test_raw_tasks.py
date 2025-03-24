import json
import os
import shutil
from pathlib import Path
from contextlib import nullcontext

import pytest

# Import the classes under test. Adjust the import path as needed.
from app.raw_tasks import RawSweTask, RawGithubTask, RawLocalTask
from app.task import PlainTask, SweTask
from app import utils as app_utils

###############################################################################
# Tests for RawSweTask
###############################################################################


@pytest.fixture
def dummy_swe_task_data():
    task_id = "task123"
    setup_info = {
        "repo_path": "/dummy/repo",
        "env_name": "dummy_env",
        "pre_install": "echo pre_install",
        "install": "echo install",
        "test_cmd": "pytest",
    }
    task_info = {
        "base_commit": "abc123",
        "hints_text": "dummy hints",
        "created_at": "2023-01-01",
        "test_patch": "dummy test patch",
        "repo": "dummy_repo",
        "problem_statement": "This is a dummy problem statement.",
        "version": "1.0",
        "instance_id": "instance123",
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
        "patch": "dummy developer patch",
    }
    return task_id, setup_info, task_info


def test_raw_swe_task_to_task_and_dump_meta(tmp_path, dummy_swe_task_data):
    """
    Verify that RawSweTask:
      - Returns the correct task_id.
      - Converts to a SweTask with expected attributes.
      - dump_meta_data writes meta.json, problem_statement.txt, and developer_patch.diff.
    """
    task_id, setup_info, task_info = dummy_swe_task_data
    raw_task = RawSweTask(task_id, setup_info, task_info)

    # Test task_id property.
    assert raw_task.task_id == task_id

    # Test to_task() returns a SweTask with expected attributes.
    task_obj = raw_task.to_task()
    assert isinstance(task_obj, SweTask)
    assert task_obj.problem_statement == task_info["problem_statement"]
    assert task_obj.repo_path == setup_info["repo_path"]

    # Test dump_meta_data writes expected files.
    output_dir = tmp_path / "swe_output"
    output_dir.mkdir()
    raw_task.dump_meta_data(str(output_dir))

    meta_file = output_dir / "meta.json"
    ps_file = output_dir / "problem_statement.txt"
    patch_file = output_dir / "developer_patch.diff"

    assert meta_file.exists()
    assert ps_file.exists()
    assert patch_file.exists()

    meta = json.loads(meta_file.read_text())
    assert meta["task_id"] == task_id
    assert meta["setup_info"] == setup_info
    assert meta["task_info"] == task_info

    assert ps_file.read_text() == task_info["problem_statement"]
    assert patch_file.read_text() == task_info["patch"]


###############################################################################
# Tests for RawGithubTask
###############################################################################
def test_raw_github_task_to_task_and_dump_meta(monkeypatch, tmp_path):
    """
    Verify that RawGithubTask:
      - Sets problem_statement and created_at based on fetched issue.
      - dump_meta_data writes meta.json with expected keys.
      - to_task() returns a PlainTask with expected attributes.
    """
    # Dummy parameters.
    task_id = "gh123"
    clone_link = "https://github.com/dummy/dummy_repo.git"
    commit_hash = "dummy_commit"
    issue_link = "https://github.com/dummy/dummy_repo/issues/42"
    setup_dir = str(tmp_path / "github_setup")
    os.makedirs(setup_dir, exist_ok=True)

    # Monkeypatch fetch_github_issue to return dummy data.
    def dummy_fetch_github_issue(cls, issue_url):
        return ("Dummy Title", "Dummy body", "2023-01-02")

    monkeypatch.setattr(
        RawGithubTask, "fetch_github_issue", classmethod(dummy_fetch_github_issue)
    )

    # Monkeypatch clone_repo to do nothing.
    monkeypatch.setattr(RawGithubTask, "clone_repo", lambda self: None)

    raw_task = RawGithubTask(task_id, clone_link, commit_hash, issue_link, setup_dir)
    # Expect problem_statement to be "Dummy Title\nDummy body"
    assert raw_task.problem_statement == "Dummy Title\nDummy body"
    assert raw_task.created_at == "2023-01-02"

    # Test dump_meta_data.
    output_dir = tmp_path / "github_output"
    output_dir.mkdir()
    raw_task.dump_meta_data(str(output_dir))
    meta_file = Path(output_dir, "meta.json")
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["task_info"]["base_commit"] == commit_hash

    # Test to_task returns a PlainTask.
    task_obj = raw_task.to_task()
    from app.task import PlainTask

    assert isinstance(task_obj, PlainTask)
    assert task_obj.problem_statement == raw_task.problem_statement
    assert task_obj.local_path == raw_task.clone_path


###############################################################################
# Tests for RawLocalTask
###############################################################################
def test_raw_local_task_to_task_and_dump_meta(monkeypatch, tmp_path):
    """
    Verify that RawLocalTask:
      - Initializes the repo using monkeypatched app_utils methods (without actually running git).
      - Reads the issue file.
      - dump_meta_data writes meta.json with expected keys.
      - to_task() returns a PlainTask with correct attributes.
    """
    # Create dummy local repo directory and issue file.
    local_repo = str(tmp_path / "local_repo")
    os.makedirs(local_repo, exist_ok=True)
    issue_file = tmp_path / "issue.txt"
    issue_file.write_text("Local issue content")

    task_id = "local123"
    # Monkeypatch git-related functions so that no actual git is executed.
    from app import utils as app_utils

    monkeypatch.setattr(app_utils, "cd", lambda path: nullcontext())
    monkeypatch.setattr(app_utils, "is_git_repo", lambda: True)
    monkeypatch.setattr(app_utils, "initialize_git_repo_and_commit", lambda: None)
    monkeypatch.setattr(app_utils, "get_current_commit_hash", lambda: "dummy_commit")

    raw_task = RawLocalTask(task_id, local_repo, str(issue_file))

    # Test that init_local_repo returns our dummy commit.
    commit = raw_task.init_local_repo()
    assert commit == "dummy_commit"

    # Test read_issue_from_file.
    issue_content = raw_task.read_issue_from_file()
    assert issue_content == "Local issue content"

    # Test dump_meta_data.
    output_dir = tmp_path / "local_output"
    output_dir.mkdir()
    raw_task.dump_meta_data(str(output_dir))
    meta_file = Path(output_dir, "meta.json")
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["task_info"]["base_commit"] == "dummy_commit"
    assert meta["task_info"]["problem_statement"] == "Local issue content"

    # Test to_task returns a PlainTask with expected attributes.
    task_obj = raw_task.to_task()
    from app.task import PlainTask

    assert isinstance(task_obj, PlainTask)
    assert task_obj.problem_statement == "Local issue content"
    assert task_obj.local_path == local_repo
