import ast
import contextlib
import glob
import json
import os
import shutil
import subprocess
from functools import wraps
from os.path import join as pjoin
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

import pytest

import app.utils as utils


# -----------------------------------------------------------------------------
# Test cd: verify that we change directory and then revert.
# -----------------------------------------------------------------------------
def test_cd(tmp_path):
    original_dir = os.getcwd()
    new_dir = tmp_path / "new_dir"
    new_dir.mkdir()
    with utils.cd(str(new_dir)):
        assert os.getcwd() == str(new_dir)
    # After the context, we should be back to the original directory.
    assert os.getcwd() == original_dir


# -----------------------------------------------------------------------------
# Test run_command (success)
# -----------------------------------------------------------------------------
def test_run_command_success(monkeypatch):
    # Fake subprocess.run to return a dummy CompletedProcess.
    def fake_run(cmd, **kwargs):
        return CompletedProcess(cmd, 0, stdout="success", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    cp = utils.run_command(["dummy", "command"], text=True)
    assert cp.stdout == "success"


# -----------------------------------------------------------------------------
# Test run_command (failure)
# -----------------------------------------------------------------------------
def test_run_command_failure(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise CalledProcessError(1, cmd, output="fail", stderr="error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(CalledProcessError):
        utils.run_command(["dummy", "command"], text=True)


# -----------------------------------------------------------------------------
# Test is_git_repo
# -----------------------------------------------------------------------------
def test_is_git_repo(tmp_path):
    # Create a temporary directory with a .git folder.
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    with utils.cd(str(repo_dir)):
        assert utils.is_git_repo() is True
    # In a directory without .git, it should return False.
    nonrepo = tmp_path / "not_repo"
    nonrepo.mkdir()
    with utils.cd(str(nonrepo)):
        assert utils.is_git_repo() is False


# -----------------------------------------------------------------------------
# Test initialize_git_repo_and_commit
# -----------------------------------------------------------------------------
def test_initialize_git_repo_and_commit(monkeypatch, tmp_path):
    calls = []

    def fake_run_command(cmd, **kwargs):
        calls.append(cmd)
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(utils, "run_command", fake_run_command)
    d = tmp_path / "repo_init"
    d.mkdir()
    with utils.cd(str(d)):
        utils.initialize_git_repo_and_commit()
    expected = [
        ["git", "init"],
        ["git", "add", "."],
        ["git", "commit", "-m", "Temp commit made by ACR."],
    ]
    assert calls == expected


# -----------------------------------------------------------------------------
# Test get_current_commit_hash
# -----------------------------------------------------------------------------
def test_get_current_commit_hash(monkeypatch):
    def fake_run(cmd, **kwargs):
        return CompletedProcess(cmd, 0, stdout="dummyhash\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    commit = utils.get_current_commit_hash()
    assert commit == "dummyhash"


# -----------------------------------------------------------------------------
# Test repo_commit_current_changes
# -----------------------------------------------------------------------------
def test_repo_commit_current_changes(monkeypatch):
    calls = []

    def fake_run_command(cmd, **kwargs):
        calls.append(cmd)
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(utils, "run_command", fake_run_command)
    utils.repo_commit_current_changes()
    expected = [
        ["git", "add", "."],
        [
            "git",
            "commit",
            "--allow-empty",
            "-m",
            "Temporary commit for storing changes",
        ],
    ]
    assert calls == expected


# -----------------------------------------------------------------------------
# Test clone_repo
# -----------------------------------------------------------------------------
def test_clone_repo(monkeypatch, tmp_path):
    calls = []

    def fake_run_command(cmd, **kwargs):
        calls.append(cmd)
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(utils, "run_command", fake_run_command)
    dest_dir = str(tmp_path / "dest")
    cloned_name = "cloned_repo"
    # Ensure destination directory exists.
    utils.create_dir_if_not_exists(dest_dir)
    # Patch cd to do nothing.
    monkeypatch.setattr(utils, "cd", lambda newdir: contextlib.nullcontext())
    cloned_dir = utils.clone_repo("dummy_link", dest_dir, cloned_name)
    assert cloned_dir == os.path.join(dest_dir, cloned_name)
    assert any("clone" in " ".join(cmd) for cmd in calls)


# -----------------------------------------------------------------------------
# Test clone_repo_and_checkout
# -----------------------------------------------------------------------------
def test_clone_repo_and_checkout(monkeypatch, tmp_path):
    calls = []

    def fake_run_command(cmd, **kwargs):
        calls.append(cmd)
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(utils, "run_command", fake_run_command)
    dest_dir = str(tmp_path / "dest")
    cloned_name = "cloned_repo"
    monkeypatch.setattr(utils, "cd", lambda newdir: contextlib.nullcontext())
    cloned_dir = utils.clone_repo_and_checkout(
        "dummy_link", "dummy_commit", dest_dir, cloned_name
    )
    assert any("checkout" in " ".join(cmd) for cmd in calls)
    assert cloned_dir == os.path.join(dest_dir, cloned_name)


# -----------------------------------------------------------------------------
# Test repo_clean_changes
# -----------------------------------------------------------------------------
def test_repo_clean_changes(monkeypatch):
    calls = []

    def fake_run_command(cmd, **kwargs):
        calls.append(cmd)
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(utils, "run_command", fake_run_command)
    utils.repo_clean_changes()
    expected = [["git", "reset", "--hard"], ["git", "clean", "-fd"]]
    assert calls == expected


# -----------------------------------------------------------------------------
# Test run_script_in_conda
# -----------------------------------------------------------------------------
def test_run_script_in_conda(monkeypatch):
    def fake_run(cmd, **kwargs):
        return CompletedProcess(cmd, 0, stdout="script output", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    cp = utils.run_script_in_conda(
        ["script.py"], "dummy_env", text=True, capture_output=True
    )
    cmd_str = " ".join(cp.args)
    assert "conda run -n dummy_env python" in cmd_str
    assert cp.stdout == "script output"


# -----------------------------------------------------------------------------
# Test create_dir_if_not_exists
# -----------------------------------------------------------------------------
def test_create_dir_if_not_exists(tmp_path):
    d = tmp_path / "new_dir"
    if d.exists():
        shutil.rmtree(str(d))
    utils.create_dir_if_not_exists(str(d))
    assert d.exists()


# -----------------------------------------------------------------------------
# Test create_fresh_dir
# -----------------------------------------------------------------------------
def test_create_fresh_dir(tmp_path):
    d = tmp_path / "fresh_dir"
    d.mkdir()
    (d / "dummy.txt").write_text("data")
    utils.create_fresh_dir(str(d))
    assert d.exists()
    assert list(d.iterdir()) == []


# -----------------------------------------------------------------------------
# Test to_relative_path and to_absolute_path
# -----------------------------------------------------------------------------
def test_relative_and_absolute_paths(tmp_path):
    project_root = str(tmp_path / "project")
    os.makedirs(project_root, exist_ok=True)
    abs_path = os.path.join(project_root, "subdir", "file.txt")
    os.makedirs(os.path.join(project_root, "subdir"), exist_ok=True)
    with open(abs_path, "w") as f:
        f.write("data")
    rel = utils.to_relative_path(abs_path, project_root)
    assert rel == os.path.join("subdir", "file.txt")
    rel2 = utils.to_relative_path("subdir/file.txt", project_root)
    assert rel2 == "subdir/file.txt"
    abs_again = utils.to_absolute_path("subdir/file.txt", project_root)
    assert abs_again == os.path.join(project_root, "subdir/file.txt")


# -----------------------------------------------------------------------------
# Test find_file
# -----------------------------------------------------------------------------
def test_find_file(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    file1 = base / "file.txt"
    file1.write_text("data")
    sub = base / "sub"
    sub.mkdir()
    file2 = sub / "file.txt"
    file2.write_text("data")
    found = utils.find_file(str(base), "file.txt")
    assert found in ["file.txt", os.path.join("sub", "file.txt")]
    found2 = utils.find_file(str(base), "file.txt")
    assert found2 is not None
    found3 = utils.find_file(str(base), os.path.join("sub", "file.txt"))
    assert found3 == os.path.join("sub", "file.txt")
    assert utils.find_file(str(base), "nofile.txt") is None


# -----------------------------------------------------------------------------
# Test parse_function_invocation
# -----------------------------------------------------------------------------
def test_parse_function_invocation():
    invocation = "my_func('arg1', 2, \"arg3\")"
    func_name, args = utils.parse_function_invocation(invocation)
    assert func_name == "my_func"
    # The arguments may be returned as strings.
    assert "arg1" in args
    # Check that one argument representing 2 is present (as "2" or 2 converted to string).
    assert any(arg == "2" or arg == str(2) for arg in args)
    assert "arg3" in args


# -----------------------------------------------------------------------------
# Test catch_all_and_log
# -----------------------------------------------------------------------------
def test_catch_all_and_log(monkeypatch):
    def faulty():
        raise ValueError("error occurred")

    decorated = utils.catch_all_and_log(faulty)
    result = decorated()
    assert isinstance(result, tuple)
    assert result[2] is False
    assert "error occurred" in result[0]


# -----------------------------------------------------------------------------
# Test coroutine decorator
# -----------------------------------------------------------------------------
def test_coroutine():
    @utils.coroutine
    def gen():
        x = yield
        yield x + 1

    c = gen()
    result = c.send(1)
    assert result == 2
