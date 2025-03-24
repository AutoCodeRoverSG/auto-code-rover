import json
import os
import shutil
import subprocess
from pathlib import Path
import pytest

import app.post_process as pp
from app.post_process import ExtractStatus


# =============================================================================
# Test for count_and_organize_tasks
# =============================================================================
def test_count_and_organize_tasks(tmp_path):
    # Create a fake experiment directory with two subdirectories.
    expr_dir = tmp_path / "expr"
    expr_dir.mkdir()
    # Create subdirectories simulating individual experiment results.
    task1_dir = expr_dir / "task1_result"
    task1_dir.mkdir()
    other_dir = expr_dir / "other_result"
    other_dir.mkdir()

    # We want to move those dirs whose names start with an element in task_list.
    task_list = ["task1"]
    task_list_name = "category1"
    task_exp_names = [d.name for d in [task1_dir, other_dir]]
    message = pp.count_and_organize_tasks(
        task_list, task_list_name, task_exp_names, str(expr_dir)
    )

    # Check that the message contains expected counts and task id.
    assert "Total number of tasks in category1: 1/2" in message
    assert "task1" in message

    # Check that a new directory named "category1" was created.
    new_dir = expr_dir / "category1"
    assert new_dir.exists()
    # task1_result should have been moved to new_dir.
    assert not (expr_dir / "task1_result").exists()
    # other_result remains in the original experiment directory.
    assert (expr_dir / "other_result").exists()


# =============================================================================
# Tests for record_extract_status and read_extract_status
# =============================================================================
def test_record_and_read_extract_status(tmp_path):
    indiv_dir = tmp_path / "indiv"
    indiv_dir.mkdir()

    # Record a status for the first time.
    pp.record_extract_status(str(indiv_dir), ExtractStatus.APPLICABLE_PATCH)
    record_file = indiv_dir / "extract_status.json"
    assert record_file.exists()
    data = json.loads(record_file.read_text())
    # (The enum is dumped as its value.)
    assert data["extract_status"] == [ExtractStatus.APPLICABLE_PATCH.value]

    # Record another status.
    pp.record_extract_status(str(indiv_dir), ExtractStatus.MATCHED_BUT_EMPTY_DIFF)
    data = json.loads(record_file.read_text())
    assert len(data["extract_status"]) == 2

    # Test read_extract_status: It will glob for extract_status.json files.
    best_status, best_file = pp.read_extract_status(str(indiv_dir))
    # Given two statuses, the best (highest) is APPLICABLE_PATCH.
    assert best_status == ExtractStatus.APPLICABLE_PATCH
    # With one record file and two statuses, best_idx is computed as 0.
    expected_best_file = record_file.with_name("extracted_patch_0.diff")
    assert str(best_file) == str(expected_best_file)


# =============================================================================
# Test for get_final_patch_path
# =============================================================================
def test_get_final_patch_path(tmp_path):
    indiv_dir = tmp_path / "indiv"
    indiv_dir.mkdir()
    record_file = indiv_dir / "extract_status.json"
    # Write a record with a good status.
    json.dump(
        {"extract_status": [ExtractStatus.APPLICABLE_PATCH.value]},
        record_file.open("w"),
        indent=4,
    )
    final_path = pp.get_final_patch_path(str(indiv_dir))
    assert final_path is not None

    # Now write a record with NO_PATCH.
    record_file.write_text(
        json.dumps({"extract_status": [ExtractStatus.NO_PATCH.value]}, indent=4)
    )
    final_path = pp.get_final_patch_path(str(indiv_dir))
    assert final_path is None


# =============================================================================
# Test for is_valid_json
# =============================================================================
def test_is_valid_json():
    valid_str = '{"key": "value"}'
    status, data = pp.is_valid_json(valid_str)
    assert status == ExtractStatus.IS_VALID_JSON
    assert data == {"key": "value"}

    invalid_str = '{"key": "value"'
    status, data = pp.is_valid_json(invalid_str)
    assert status == ExtractStatus.NOT_VALID_JSON
    assert data is None


# =============================================================================
# Test for convert_response_to_diff
# =============================================================================
def test_convert_response_to_diff(monkeypatch, tmp_path):
    import app.post_process as pp
    from app.post_process import ExtractStatus

    # Create a fake task directory with a meta.json.
    task_dir = tmp_path / "task_dir"
    task_dir.mkdir()
    meta = {
        "task_info": {"base_commit": "dummy_commit"},
        "setup_info": {"repo_path": str(task_dir)},
    }
    (task_dir / "meta.json").write_text(json.dumps(meta))

    # Dummy raw patch response.
    response = "dummy patch content"

    # Create a dummy edit object.
    class DummyEdit:
        def __init__(self):
            self.filename = "dummy_file.txt"
            self.before = "old code"

    dummy_edit = DummyEdit()

    # Patch parse_edits to return a list with our dummy edit.
    monkeypatch.setattr(pp, "parse_edits", lambda patch: [dummy_edit])
    # Patch is_test_file to always return False.
    monkeypatch.setattr(pp, "is_test_file", lambda filename: False)
    # Patch repo_clean_changes to do nothing.
    monkeypatch.setattr(pp.apputils, "repo_clean_changes", lambda: None)

    # Patch run_command to return a dummy diff.
    class DummyCompletedProcess:
        def __init__(self, text):
            self.stdout = text.encode("utf-8")

    monkeypatch.setattr(
        pp.apputils,
        "run_command",
        lambda cmd, **kwargs: DummyCompletedProcess("dummy diff"),
    )
    # Patch repo_reset_and_clean_checkout to do nothing.
    monkeypatch.setattr(
        pp.apputils, "repo_reset_and_clean_checkout", lambda commit: None
    )
    # Patch find_file to always "find" the dummy file.
    monkeypatch.setattr(
        pp.apputils, "find_file", lambda repo, target: str(Path(repo) / target)
    )
    # Patch apply_edit to simulate a successful edit application.
    monkeypatch.setattr(pp, "apply_edit", lambda edit, found_file: found_file)

    status, summary, diff = pp.convert_response_to_diff(
        response, str(task_dir), standalone_mode=True
    )
    # Expect a successful extraction.
    assert (
        status == ExtractStatus.APPLICABLE_PATCH
    ), f"Expected APPLICABLE_PATCH, got {status}"
    assert diff == "dummy diff"


# =============================================================================
# Test for extract_diff_one_instance
# =============================================================================
def test_extract_diff_one_instance(tmp_path):
    # When the raw patch file does not exist.
    raw_patch_file = str(tmp_path / "nonexistent.patch")
    extracted_file = str(tmp_path / "extracted.diff")
    status, summary = pp.extract_diff_one_instance(
        raw_patch_file, extracted_file, standalone_mode=True
    )
    assert status == ExtractStatus.NO_PATCH
    assert "No raw patch file" in summary


# =============================================================================
# Test for organize_experiment_results
# =============================================================================
def test_organize_experiment_results(tmp_path):
    # Create an experiment directory with one task dir (its name must contain "__").
    expr_dir = tmp_path / "expr"
    expr_dir.mkdir()
    task_dir = expr_dir / "task__1"
    task_dir.mkdir()
    # Create a minimal meta.json and extract_status.json in the task dir.
    meta = {"task_id": "task1"}
    (task_dir / "meta.json").write_text(json.dumps(meta))
    record = {"extract_status": [ExtractStatus.APPLICABLE_PATCH.value]}
    (task_dir / "extract_status.json").write_text(json.dumps(record, indent=4))

    pp.organize_experiment_results(str(expr_dir))
    # The target folder is derived from the status value (lowercase).
    category_dir = expr_dir / ExtractStatus.APPLICABLE_PATCH.value.lower()
    moved_task_dir = category_dir / task_dir.name
    assert moved_task_dir.exists()
    assert not task_dir.exists()


# =============================================================================
# Test for extract_swe_bench_input
# =============================================================================
def test_extract_swe_bench_input(tmp_path, monkeypatch):
    # Create a fake experiment directory with an "applicable_patch" folder.
    expr_dir = tmp_path / "expr"
    expr_dir.mkdir()
    applicable_dir = expr_dir / "applicable_patch"
    applicable_dir.mkdir()
    task_dir = applicable_dir / "task1"
    task_dir.mkdir()
    # Create a meta.json file with a task_id.
    meta = {"task_id": "task1"}
    (task_dir / "meta.json").write_text(json.dumps(meta))
    # Patch common.SELECTED_MODEL so that its name is "dummy-model".
    from app import model

    dummy_model = type("DummyModel", (), {"name": "dummy-model"})
    model.common.SELECTED_MODEL = dummy_model
    # Create a selected_patch.json file with a chosen patch name.
    selected_patch = {"selected_patch": "extracted.diff"}
    (task_dir / "selected_patch.json").write_text(json.dumps(selected_patch))
    # Create the extracted diff file.
    diff_file = task_dir / "extracted.diff"
    diff_file.write_text("dummy diff content")

    swe_input_file = pp.extract_swe_bench_input(str(expr_dir))
    predictions_file = Path(swe_input_file)
    assert predictions_file.exists()
    data = json.loads(predictions_file.read_text())
    assert len(data) == 1
    result = data[0]
    assert result["instance_id"] == "task1"
    assert result["model_patch"] == "dummy diff content"
    assert result["model_name_or_path"] == "dummy-model"


# =============================================================================
# Test for extract_organize_and_form_input (wrapper)
# =============================================================================
def test_extract_organize_and_form_input(tmp_path, monkeypatch):
    # Create a fake experiment directory with a dummy task directory.
    expr_dir = tmp_path / "expr"
    expr_dir.mkdir()
    task_dir = expr_dir / "task__1"
    task_dir.mkdir()
    # Create meta.json (simulate a task)
    meta = {
        "task_id": "task1",
        "task_info": {"base_commit": "dummy_commit"},
        "setup_info": {"repo_path": str(task_dir)},
    }
    (task_dir / "meta.json").write_text(json.dumps(meta))
    # Create a dummy raw patch file.
    raw_patch_file = task_dir / "agent_patch_raw_1"
    raw_patch_file.write_text("dummy patch content")
    # Patch functions to bypass extraction (simulate successful extraction).
    monkeypatch.setattr(
        pp,
        "extract_diff_one_instance",
        lambda raw, ext, standalone_mode=False: (ExtractStatus.APPLICABLE_PATCH, ""),
    )
    monkeypatch.setattr(pp, "record_extract_status", lambda d, s: None)
    monkeypatch.setattr(pp, "organize_experiment_results", lambda d: None)
    # Patch extract_swe_bench_input to simply return a dummy file path.
    monkeypatch.setattr(
        pp,
        "extract_swe_bench_input",
        lambda d: os.path.join(d, "predictions_for_swebench.json"),
    )

    pp.extract_organize_and_form_input(str(expr_dir))
    predictions_file = Path(expr_dir) / "predictions_for_swebench.json"
    # Simulate file creation.
    predictions_file.write_text("dummy predictions")
    assert predictions_file.exists()
    assert predictions_file.read_text() == "dummy predictions"


# =============================================================================
# Test for organize_and_form_input (wrapper)
# =============================================================================
def test_organize_and_form_input(tmp_path, monkeypatch):
    expr_dir = tmp_path / "expr"
    expr_dir.mkdir()
    # Patch organize_experiment_results and extract_swe_bench_input.
    monkeypatch.setattr(pp, "organize_experiment_results", lambda d: None)
    dummy_output = os.path.join(str(expr_dir), "predictions_for_swebench.json")
    monkeypatch.setattr(pp, "extract_swe_bench_input", lambda d: dummy_output)
    # Create a dummy predictions file.
    predictions_file = Path(dummy_output)
    predictions_file.write_text("dummy predictions")
    result = pp.organize_and_form_input(str(expr_dir))
    assert result == dummy_output
    assert predictions_file.read_text() == "dummy predictions"
