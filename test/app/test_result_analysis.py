import json
from glob import glob
from os.path import join as pjoin
from pathlib import Path
from pprint import pprint
from contextlib import contextmanager

import pytest
import emojis

from app.result_analysis import (
    analyze,
    analyze_one_task,
    get_resolved,
    get_instance_names_from_dir,
)
import app.utils as apputils


###############################################################################
# Define a dummy context manager for apputils.cd
###############################################################################
@contextmanager
def dummy_cd(path):
    yield


###############################################################################
# DummyModel to simulate model.call responses (if needed in future tests)
###############################################################################
class DummyModel:
    def __init__(self, responses):
        self.responses = responses  # list of response strings
        self.call_count = 0

    def setup(self):
        pass

    def call(self, messages, **kwargs):
        response = self.responses[self.call_count]
        self.call_count += 1
        return (response,)


###############################################################################
# Test for get_instance_names_from_dir
###############################################################################
def test_get_instance_names_from_dir(tmp_path):
    """
    Create dummy subdirectories with names following the pattern
    <instance>_<...>_<...> and verify that the instance names
    are correctly extracted.
    """
    # Create directories such as "taskA_extra_001", "taskB_info_002", "taskC_data_003"
    (tmp_path / "taskA_extra_001").mkdir()
    (tmp_path / "taskB_info_002").mkdir()
    (tmp_path / "taskC_data_003").mkdir()
    names = get_instance_names_from_dir(tmp_path)
    assert set(names) == {"taskA", "taskB", "taskC"}


###############################################################################
# Test for get_resolved
###############################################################################
def test_get_resolved(tmp_path):
    """
    Create a temporary report directory with a report.json file containing a list
    of resolved instances and verify that get_resolved returns the sorted list.
    """
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    data = {"resolved": ["taskA", "taskC"]}
    report_file = report_dir / "report.json"
    report_file.write_text(json.dumps(data))
    resolved = get_resolved(tmp_path)
    assert resolved == sorted(["taskA", "taskC"])


###############################################################################
# Test analyze_one_task
###############################################################################
def test_analyze_one_task(tmp_path, monkeypatch):
    """
    Create a temporary task instance directory with:
      - Two extracted patch files (e.g. "extracted_patch_1.diff" and "extracted_patch_2.diff"),
        where the latter should be chosen.
      - A developer patch file.
      - A meta.json file with necessary information.
    Monkeypatch apputils.cd and repo_reset_and_clean_checkout to do nothing.
    Monkeypatch compare_fix_locations to record the model_patch argument.
    Verify that analyze_one_task returns the expected dummy tuple and that the chosen
    model_patch is the one with the highest number.
    """
    task_dir = tmp_path / "task_instance"
    task_dir.mkdir()
    # Create two extracted patch files.
    patch1 = task_dir / "extracted_patch_1.diff"
    patch1.write_text("content1")
    patch2 = task_dir / "extracted_patch_2.diff"
    patch2.write_text("content2")
    # Create developer patch.
    dev_patch = task_dir / "developer_patch.diff"
    dev_patch.write_text("dev content")
    # Create meta.json.
    meta = {
        "setup_info": {"repo_path": str(tmp_path / "dummy_repo")},
        "task_info": {"base_commit": "dummy_commit"},
    }
    meta_file = task_dir / "meta.json"
    meta_file.write_text(json.dumps(meta))

    # Monkeypatch apputils.cd and repo_reset_and_clean_checkout to do nothing.
    monkeypatch.setattr(apputils, "cd", dummy_cd)
    monkeypatch.setattr(apputils, "repo_reset_and_clean_checkout", lambda commit: None)

    # Record the model_patch argument passed to compare_fix_locations.
    recorded = {}

    def dummy_compare_fix_locations(model_patch, dev_patch_arg, project_path):
        recorded["model_patch"] = model_patch
        return ("dummy_model_extra", True, "dummy_dev_extra")

    # Override compare_fix_locations in app.result_analysis.
    monkeypatch.setattr(
        "app.result_analysis.compare_fix_locations", dummy_compare_fix_locations
    )

    result = analyze_one_task(str(task_dir))
    assert result == ("dummy_model_extra", True, "dummy_dev_extra")
    # The expected model_patch is the one with the highest suffix number.
    expected_model_patch = str(patch2)
    assert recorded.get("model_patch") == expected_model_patch


###############################################################################
# Test analyze
###############################################################################
def test_analyze(tmp_path, monkeypatch, capsys):
    """
    Create a temporary directory structure for a SWE-bench experiment that includes:
      - Directories: applicable_patch, raw_patch_but_unmatched, raw_patch_but_unparsed, no_patch, and report.
      - In each, create one dummy instance directory.
      - A report.json file listing some resolved instances.
    Monkeypatch analyze_one_task to return a dummy tuple.
    Override emojis.encode to return its input (ensuring expected emoji text is present).
    Run analyze() and capture its printed output, verifying that it contains expected substrings.
    """
    expr_dir = tmp_path / "expr"
    expr_dir.mkdir()
    # Create required subdirectories.
    (expr_dir / "applicable_patch").mkdir()
    (expr_dir / "raw_patch_but_unmatched").mkdir()
    (expr_dir / "raw_patch_but_unparsed").mkdir()
    (expr_dir / "no_patch").mkdir()
    report_dir = expr_dir / "report"
    report_dir.mkdir()

    # Create one dummy instance in each.
    (expr_dir / "applicable_patch" / "taskA_extra_001").mkdir()
    (expr_dir / "raw_patch_but_unmatched" / "taskB_extra_002").mkdir()
    (expr_dir / "raw_patch_but_unparsed" / "taskC_extra_003").mkdir()
    (expr_dir / "no_patch" / "taskD_extra_004").mkdir()

    # Create report.json with resolved instances.
    report_data = {"resolved": ["taskA", "taskC"]}
    (report_dir / "report.json").write_text(json.dumps(report_data))

    # Monkeypatch analyze_one_task to return a dummy tuple.
    monkeypatch.setattr(
        "app.result_analysis.analyze_one_task",
        lambda task_expr_dir: ("dummy_model_extra", True, "dummy_dev_extra"),
    )

    # Override emojis.encode so that it returns its input unchanged.
    import emojis

    monkeypatch.setattr(emojis, "encode", lambda s: s)

    # Run analyze and capture output.
    analyze(str(expr_dir))
    captured = capsys.readouterr().out
    # Check that output contains expected key phrases.
    assert "Total instances:" in captured
    assert "No patch at all:" in captured
    assert "Unparsed:" in captured
    assert "Unmatched:" in captured
    assert "Applicable but not resolved:" in captured
    # Check that output contains at least one expected emoji indicator.
    lower_out = captured.lower()
    assert ":thumbsup:" in lower_out or ":collision:" in lower_out
