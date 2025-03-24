import json
import os
import shutil
from pathlib import Path
import tempfile

import pytest

from app.manage import ProjectApiManager
from app.analysis.sbfl import NoCoverageData
from app.task import Task
from app.data_structures import MessageThread
from test.pytest_utils import *


###############################################################################
# Test fault_localization: success path
###############################################################################
def test_fault_localization_success(monkeypatch, tmp_path):
    """
    Simulate a successful fault localization.
    Monkeypatch sbfl.run, sbfl.collate_results, and sbfl.map_collated_results_to_methods
    to return dummy data. Also, create a dummy log file that is moved.
    Verify that fault_localization returns a tuple with a tool_output,
    a summary, and True.
    """
    # Create dummy file names and ranked lines.
    dummy_test_files = ["dummy1.py", "dummy2.py"]
    dummy_ranked_lines = [("dummy/path/dummy1.py", 10, 20, "score")]
    # Dummy log file path.
    dummy_log_file = tmp_path / "dummy_log.txt"
    dummy_log_file.write_text("log content")

    # Dummy ranked ranges and methods (absolute)
    dummy_ranked_ranges_abs = [("dummy/path/dummy1.py", "", "", "score")]
    dummy_ranked_methods_abs = [
        ("dummy/path/dummy1.py", "DummyClass", "dummy_method", "score")
    ]

    # Monkeypatch sbfl.run to return our dummy values.
    def dummy_sbfl_run(task):
        return dummy_test_files, dummy_ranked_lines, str(dummy_log_file)

    monkeypatch.setattr("app.manage.sbfl.run", dummy_sbfl_run)

    # Monkeypatch sbfl.collate_results and sbfl.map_collated_results_to_methods.
    monkeypatch.setattr(
        "app.manage.sbfl.collate_results", lambda lines, files: dummy_ranked_ranges_abs
    )
    monkeypatch.setattr(
        "app.manage.sbfl.map_collated_results_to_methods",
        lambda ranges: dummy_ranked_methods_abs,
    )

    # Monkeypatch shutil.move to simply rename the file.
    monkeypatch.setattr(shutil, "move", lambda src, dst: os.rename(src, dst))

    # Create a temporary output directory.
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create a dummy project directory.
    project_dir = tmp_path / "dummy_project"
    project_dir.mkdir()

    # Create a dummy Task.
    task = DummyTask(project_path=str(project_dir), issue="Test issue for localization")

    manager = ProjectApiManager(task, str(output_dir))

    # Call fault_localization.
    tool_output, summary, flag = manager.fault_localization()

    # Check that a non-empty tool_output and summary are returned, and flag is True.
    assert flag is True
    assert "Top-" in tool_output
    assert "suspicious methods" in summary

    # Also verify that the result files were written.
    sbfl_result_file = output_dir / "sbfl_result.json"
    sbfl_method_result_file = output_dir / "sbfl_result_method.json"
    assert sbfl_result_file.exists()
    assert sbfl_method_result_file.exists()

    # Check that the log file was moved.
    moved_log = output_dir / "run_developer_tests.log"
    assert moved_log.exists()


###############################################################################
# Test fault_localization: no coverage data path
###############################################################################
def test_fault_localization_no_coverage(monkeypatch, tmp_path):
    """
    Simulate a case where sbfl.run raises a NoCoverageData exception.
    Verify that fault_localization writes empty output files and returns an error message.
    """
    dummy_log_file = tmp_path / "dummy_log.txt"
    dummy_log_file.write_text("log content")

    # Dummy exception that includes a testing_log_file attribute.
    class DummyNoCoverage(NoCoverageData):
        def __init__(self, message):
            super().__init__(message)
            self.testing_log_file = str(dummy_log_file)

    def dummy_sbfl_run(task):
        raise DummyNoCoverage("No coverage")

    monkeypatch.setattr("app.manage.sbfl.run", dummy_sbfl_run)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    project_dir = tmp_path / "dummy_project"
    project_dir.mkdir()
    task = DummyTask(project_path=str(project_dir), issue="Test issue")
    manager = ProjectApiManager(task, str(output_dir))

    tool_output, summary, flag = manager.fault_localization()
    # Expect error messages.
    assert flag is False
    assert "Error in running localization tool" in tool_output
    # Expect the result files to be empty.
    sbfl_result_file = output_dir / "sbfl_result.json"
    sbfl_method_result_file = output_dir / "sbfl_result_method.json"
    assert sbfl_result_file.read_text() == ""
    assert sbfl_method_result_file.read_text() == ""
    # Verify that the log file was moved.
    moved_log = output_dir / "run_developer_tests.log"
    assert moved_log.exists()


###############################################################################
# Test reproduce: success path
###############################################################################
def test_reproduce_success(monkeypatch, tmp_path):
    """
    Simulate a successful reproduction by monkeypatching agent_reproducer.generator to yield a tuple with run_ok True.
    Verify that reproduce returns the expected test content, summary, and success flag.
    """

    # Define a dummy generator function.
    def dummy_generator(issue_statement):
        yield ("dummy test content", MessageThread(), True)

    monkeypatch.setattr(
        "app.manage.agent_reproducer.generator", lambda issue: dummy_generator(issue)
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    project_dir = tmp_path / "dummy_project"
    project_dir.mkdir()
    task = DummyTask(project_path=str(project_dir), issue="Reproducer test issue")
    manager = ProjectApiManager(task, str(output_dir))

    test_content, summary, success = manager.reproduce(retries=3)
    assert success is True
    assert test_content == "dummy test content"
    assert "returned a reproducer test" in summary.lower()

    # Check that a file agent_reproducer.json was written.
    repro_file = output_dir / "agent_reproducer.json"
    assert repro_file.exists()


###############################################################################
# Test reproduce: failure path
###############################################################################
def test_reproduce_failure(monkeypatch, tmp_path):
    """
    Simulate a failure in reproduction by monkeypatching agent_reproducer.generator
    to yield only unsuccessful attempts.
    Verify that reproduce returns an empty test content, a failure summary, and success flag False.
    """

    # Define a dummy generator that never yields success.
    def dummy_generator(issue_statement):
        # Yield unsuccessful attempts with a non-empty MessageThread.
        for i in range(3):
            mt = MessageThread()
            mt.add_user(f"Attempt {i} failed")
            yield ("", mt, False)

    monkeypatch.setattr(
        "app.manage.agent_reproducer.generator", lambda issue: dummy_generator(issue)
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    project_dir = tmp_path / "dummy_project"
    project_dir.mkdir()
    task = DummyTask(project_path=str(project_dir), issue="Reproducer failure issue")
    manager = ProjectApiManager(task, str(output_dir))

    test_content, summary, success = manager.reproduce(retries=3)
    assert success is False
    assert test_content == ""
    assert "failed to write a reproducer test" in summary.lower()

    # Check that agent_reproducer.json was written.
    repro_file = output_dir / "agent_reproducer.json"
    assert repro_file.exists()
