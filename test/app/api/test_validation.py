import ast
import itertools
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
import textwrap

import pytest
from unidiff import PatchSet

from app import config
from app.api import validation
from app.api.validation import *
from app.data_structures import MethodId
from app.task import SweTask, Task
from app.agents.agent_write_patch import PatchHandle


# --- Dummy and Fixture Setup ---


class DummyTask(Task):
    def __init__(self, task_id, project_path):
        self.task_id = task_id
        self.project_path = project_path

    def validate(self, patch_content):
        # Simulate a validation that passes.
        # Create two temporary log files.
        log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        orig_log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        log_file.close()
        orig_log_file.close()
        return True, "Validation passed", log_file.name, orig_log_file.name


# DummySweTask now accepts task_id and project_path.
class DummySweTask(SweTask):
    def __init__(self, task_id, project_path):
        self.task_id = task_id
        self.project_path = str(project_path)

    def validate(self, patch_content):
        # Same dummy validate as above.
        log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        orig_log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        log_file.close()
        orig_log_file.close()
        return True, "Validation passed", log_file.name, orig_log_file.name


class DummyPatchHandle(PatchHandle):
    def __str__(self):
        return "dummy_patch"


@pytest.fixture
def tmp_project(tmp_path):
    # Create a temporary project with one dummy Python file.
    proj = tmp_path / "project"
    proj.mkdir()
    dummy_py = proj / "dummy.py"
    dummy_py.write_text(
        textwrap.dedent(
            """
            def foo():
                pass

            class Bar:
                def baz(self):
                    pass
            """
        )
    )
    return str(proj)


# Monkey-patch apputils functions used in get_changed_methods
@pytest.fixture(autouse=True)
def fake_apputils(monkeypatch):
    # Fake context manager for cd.
    class DummyCD:
        def __init__(self, directory):
            self.directory = directory

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    def fake_cd(directory):
        return DummyCD(directory)

    def fake_repo_clean_changes():
        pass

    monkeypatch.setattr("app.api.validation.apputils.cd", fake_cd)
    monkeypatch.setattr(
        "app.api.validation.apputils.repo_clean_changes", fake_repo_clean_changes
    )


# Fake subprocess.run for patch command: simulate a successful patch that changes the file.
@pytest.fixture(autouse=True)
def fake_subprocess_run(monkeypatch):
    def fake_run(cmd, cwd, stdout, stderr, text):
        # In our fake run, we simulate a patch command with returncode 0.
        class DummyCompletedProcess:
            returncode = 0
            stdout = "patch applied"
            stderr = ""

        # Additionally, we modify the file in cwd to simulate a change.
        # Expect that cmd is like: patch -p1 -f -i <diff_file>
        # We assume that in cwd, there is a file "dummy.py".
        target = Path(cwd) / "dummy.py"
        if target.is_file():
            # Replace the content to simulate a change.
            target.write_text("def foo():\n    print('changed')\n")
        return DummyCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)


# --- Tests for validation.py ---


def test_angelic_debugging_message():
    # Test that a non-empty incorrect_locations produces a message.
    incorrect_locations = [("dummy.py", MethodId("Bar", "baz"))]
    msg = angelic_debugging_message(incorrect_locations)
    assert "dummy.py" in msg
    assert "Bar" in msg
    assert "baz" in msg


def test_collect_method_definitions(tmp_path):
    # Create a temporary Python file with a function and a method.
    file_content = textwrap.dedent(
        """
        def alpha():
            pass

        class Beta:
            def gamma(self):
                pass
    """
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text(file_content)
    defs = collect_method_definitions(str(file_path))
    method_names = {mid.method_name for mid in defs.keys()}
    assert "alpha" in method_names
    assert "gamma" in method_names


def test_get_method_id(monkeypatch, tmp_path):
    # Monkey-patch method_ranges_in_file to return a fake range.
    def fake_method_ranges(file):
        return {MethodId("Dummy", "foo"): (1, 10)}

    monkeypatch.setattr("app.api.validation.method_ranges_in_file", fake_method_ranges)
    mid = get_method_id("dummy.py", 5)
    assert mid == MethodId("Dummy", "foo")
    mid_none = get_method_id("dummy.py", 20)
    assert mid_none is None


def test_get_changed_methods(tmp_project, tmp_path, monkeypatch):
    # Updated diff content with a valid hunk.
    diff_content = textwrap.dedent(
        """\
        --- a/dummy.py
        +++ b/dummy.py
        @@ -1,2 +1,2 @@
         def foo():
        -    pass
        +    print("changed")
    """
    )
    diff_file = tmp_path / "change.diff"
    diff_file.write_text(diff_content)

    def fake_collect_method_definitions(file):
        if "apply_patch_" in str(file):
            return {MethodId("", "foo"): "def foo():\n    print('changed')"}
        else:
            return {MethodId("", "foo"): "def foo():\n    pass"}

    monkeypatch.setattr(
        validation, "collect_method_definitions", fake_collect_method_definitions
    )

    result = get_changed_methods(str(diff_file), tmp_project)
    assert "dummy.py" in result
    changed_methods = result["dummy.py"]
    assert MethodId("", "foo") in changed_methods


def test_get_developer_patch_file(tmp_path, monkeypatch):
    # Create a temporary file to simulate a developer patch diff.
    task_id = "dummy_task"
    processed_data_lite = tmp_path / "processed_data_lite"
    test_dir = processed_data_lite / "test" / task_id
    test_dir.mkdir(parents=True)
    dev_patch = test_dir / "developer_patch.diff"
    dev_patch.write_text("dummy patch")

    # Monkey-patch Path.__file__ relative resolution in get_developer_patch_file.
    # Instead, we override the parent's with_name to return our tmp directory.
    def fake_with_name(self, name):
        return processed_data_lite

    monkeypatch.setattr(Path, "with_name", fake_with_name)
    path = get_developer_patch_file(task_id)
    assert Path(path).is_file()


def test_perfect_angelic_debug(monkeypatch, tmp_path):
    # Updated diff content with a valid hunk that changes dummy.py.
    diff_content = textwrap.dedent(
        """\
        --- a/dummy.py
        +++ b/dummy.py
        @@ -1,2 +1,2 @@
         def foo():
        -    pass
        +    print("changed")
    """
    )
    diff_file = tmp_path / "change.diff"
    diff_file.write_text(diff_content)

    # Create a dummy developer patch file that is a no-op (empty diff).
    # When the diff is empty, get_changed_methods returns {} for the developer patch.
    task_id = "dummy_task"
    processed_data_lite = tmp_path / "processed_data_lite"
    test_dir = processed_data_lite / "test" / task_id
    test_dir.mkdir(parents=True)
    dev_patch = test_dir / "developer_patch.diff"
    dev_patch.write_text("")  # no changes

    monkeypatch.setattr(
        validation, "get_developer_patch_file", lambda tid: str(dev_patch)
    )

    proj = tmp_path / "project"
    proj.mkdir()
    dummy_py = proj / "dummy.py"
    dummy_py.write_text("def foo():\n    pass\n")

    results = perfect_angelic_debug(task_id, str(diff_file), str(proj))
    diff_only, common, dev_only = results
    # Since the developer patch is empty, we expect the changed method to appear in diff_only.
    assert ("dummy.py", MethodId("", "foo")) in diff_only


def test_evaluate_patch(monkeypatch, tmp_path):
    config.enable_validation = True
    config.enable_perfect_angelic = True
    config.enable_angelic = False

    # Use DummySweTask with our fixed __init__.
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    dummy_py = project_dir / "dummy.py"
    dummy_py.write_text("def foo():\n    pass\n")
    task = DummySweTask(task_id="dummy_task", project_path=project_dir)
    patch_handle = DummyPatchHandle()
    patch_content = "dummy patch content"

    dummy_log = tmp_path / "dummy.log"
    dummy_log.write_text("log")
    dummy_orig_log = tmp_path / "dummy_orig.log"
    dummy_orig_log.write_text("orig log")

    def fake_validate(patch_content):
        return True, "Patch is correct", str(dummy_log), str(dummy_orig_log)

    monkeypatch.setattr(task, "validate", fake_validate)
    monkeypatch.setattr(
        validation, "perfect_angelic_debug", lambda tid, df, pp: (set(), set(), set())
    )
    monkeypatch.setattr(shutil, "move", lambda src, dst: None)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    passed, msg = evaluate_patch(task, patch_handle, patch_content, str(output_dir))
    assert passed is True
    assert "successfully resolved" in msg
