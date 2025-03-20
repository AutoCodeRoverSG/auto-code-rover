import json
import sys
from pathlib import Path
import pytest

# Global dictionary to track function calls.
call_tracker = {
    "fake_run_one_task": 0,
    "make_swe_tasks": 0,
    "group_swe_tasks_by_env": 0,
    "RawGithubTask": 0,
    "RawLocalTask": 0,
}

# Import the main module so we can patch its attributes.
import app.main as main_module
from app.main import inference

# Import your existing DummyTask from your utils module.
from test.pytest_utils import DummyTask as BaseDummyTask

# Extend the existing DummyTask to accept extra arguments without breaking behavior.
class DummyTask(BaseDummyTask):
    def __init__(self, *args, **kwargs):
        # Ignore extra arguments and use default initialization.
        super().__init__()
    
    def dump_meta_data(self, output_dir):
        # Extend DummyTask so that when dump_meta_data is called,
        # it writes a dummy meta file.
        meta_file = Path(output_dir) / "meta.json"
        meta_file.write_text('{"task": "dummy"}')

# --- Fake Implementations for Testing ---

def fake_run_one_task(task, task_output_dir, models):
    call_tracker["fake_run_one_task"] += 1
    return True

def fake_make_swe_tasks(task, task_list_file, setup_map_file, tasks_map_file):
    call_tracker["make_swe_tasks"] += 1
    return [DummyTask()]

def fake_group_swe_tasks_by_env(tasks):
    call_tracker["group_swe_tasks_by_env"] += 1
    return {"dummy_env": tasks}

def fake_RawGithubTask(*args, **kwargs):
    call_tracker["RawGithubTask"] += 1
    return DummyTask(*args, **kwargs)

def fake_RawLocalTask(*args, **kwargs):
    call_tracker["RawLocalTask"] += 1
    return DummyTask(*args, **kwargs)

def fake_organize_and_form_input(output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "swe_input.txt"
    content = "dummy input content"
    output_file.write_text(content)
    return str(output_file)

# --- Pytest Fixtures ---

@pytest.fixture(autouse=True)
def reset_call_tracker_fixture():
    call_tracker.clear()
    call_tracker.update({
        "fake_run_one_task": 0,
        "make_swe_tasks": 0,
        "group_swe_tasks_by_env": 0,
        "RawGithubTask": 0,
        "RawLocalTask": 0,
    })

@pytest.fixture(autouse=True)
def patch_functions(monkeypatch):
    # Patch the inference API call.
    monkeypatch.setattr(inference, "run_one_task", fake_run_one_task)
    # Patch functions that create tasks and perform grouping.
    monkeypatch.setattr(main_module, "make_swe_tasks", fake_make_swe_tasks)
    monkeypatch.setattr(main_module, "group_swe_tasks_by_env", fake_group_swe_tasks_by_env)
    # Leave run_task_groups unpatched so its post‐processing branch runs.
    # Patch task constructors for github and local issue commands.
    monkeypatch.setattr(main_module, "RawGithubTask", fake_RawGithubTask)
    monkeypatch.setattr(main_module, "RawLocalTask", fake_RawLocalTask)
    # Patch the post-processing function that creates the SWE input file.
    monkeypatch.setattr(main_module, "organize_and_form_input", fake_organize_and_form_input)

# --- Test Cases ---

def test_main_swe_bench(tmp_path):
    """
    Test the swe-bench command:
      - Ensure that the dummy task creation, grouping, and task-group execution functions are invoked.
      - Verify that the post-processing branch creates a swe_input.txt file with expected content.
    """
    # Create temporary dummy JSON files for setup and tasks maps.
    dummy_setup = {"dummy_task": {"env_name": "dummy_env"}}
    dummy_tasks = {"dummy_task": {}}
    setup_file = tmp_path / "setup.json"
    tasks_file = tmp_path / "tasks.json"
    setup_file.write_text(json.dumps(dummy_setup))
    tasks_file.write_text(json.dumps(dummy_tasks))

    output_dir = tmp_path / "output"

    # Prepare sys.argv as if running the "swe-bench" command.
    sys.argv = [
        "main.py",
        "swe-bench",
        "--output-dir", str(output_dir),
        "--model", "gpt-3.5-turbo-0125",
        "--model-temperature", "0.0",
        "--conv-round-limit", "15",
        "--num-processes", "1",
        "--task", "dummy_task",
        "--setup-map", str(setup_file),
        "--tasks-map", str(tasks_file),
    ]

    # Execute the main driver.
    main_module.main()

    # Assertions on fake function calls.
    assert call_tracker["make_swe_tasks"] == 1, "Expected make_swe_tasks to be called once."
    assert call_tracker["group_swe_tasks_by_env"] == 1, "Expected group_swe_tasks_by_env to be called once."

    # Check that the output file was created with expected content.
    swe_input_file = output_dir / "swe_input.txt"
    assert swe_input_file.exists(), "Expected the swe_input.txt file to be created."
    content = swe_input_file.read_text()
    assert content == "dummy input content", "Output file content does not match expected content."

def test_github_issue(tmp_path):
    """
    Test the github-issue command:
      - Verify that the patched RawGithubTask constructor is called.
    """
    output_dir = tmp_path / "output"

    sys.argv = [
        "main.py",
        "github-issue",
        "--output-dir", str(output_dir),
        "--model", "gpt-3.5-turbo-0125",
        "--model-temperature", "0.0",
        "--conv-round-limit", "15",
        "--num-processes", "1",
        "--task-id", "dummy_task",
        "--clone-link", "https://example.com/dummy.git",
        "--commit-hash", "abc123",
        "--issue-link", "https://github.com/example/repo/issues/1",
        "--setup-dir", str(tmp_path / "setup_dir"),
    ]
    main_module.main()

    # Assert that the patched RawGithubTask constructor was invoked.
    assert call_tracker["RawGithubTask"] == 1, "Expected RawGithubTask to be instantiated once."

def test_local_issue(tmp_path):
    """
    Test the local-issue command:
      - Verify that the patched RawLocalTask constructor is called.
    """
    output_dir = tmp_path / "output"

    sys.argv = [
        "main.py",
        "local-issue",
        "--output-dir", str(output_dir),
        "--model", "gpt-3.5-turbo-0125",
        "--model-temperature", "0.0",
        "--conv-round-limit", "15",
        "--num-processes", "1",
        "--task-id", "dummy_task",
        "--local-repo", str(tmp_path / "dummy_repo"),
        "--issue-file", str(tmp_path / "dummy_issue.txt"),
    ]
    main_module.main()

    # Assert that the patched RawLocalTask constructor was invoked.
    assert call_tracker["RawLocalTask"] == 1, "Expected RawLocalTask to be instantiated once."
