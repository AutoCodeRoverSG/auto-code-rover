import json
import sys
from pathlib import Path
import pytest
import os
import subprocess
import openai

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
from app.main import inference, run_raw_task
from app import config

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
    call_tracker.update(
        {
            "fake_run_one_task": 0,
            "make_swe_tasks": 0,
            "group_swe_tasks_by_env": 0,
            "RawGithubTask": 0,
            "RawLocalTask": 0,
        }
    )


@pytest.fixture(autouse=True)
def patch_functions(monkeypatch):
    # Patch the inference API call.
    monkeypatch.setattr(inference, "run_one_task", fake_run_one_task)
    # Patch functions that create tasks and perform grouping.
    monkeypatch.setattr(main_module, "make_swe_tasks", fake_make_swe_tasks)
    monkeypatch.setattr(
        main_module, "group_swe_tasks_by_env", fake_group_swe_tasks_by_env
    )
    # Leave run_task_groups unpatched so its post‐processing branch runs.
    # Patch task constructors for github and local issue commands.
    monkeypatch.setattr(main_module, "RawGithubTask", fake_RawGithubTask)
    monkeypatch.setattr(main_module, "RawLocalTask", fake_RawLocalTask)
    # Patch the post-processing function that creates the SWE input file.
    monkeypatch.setattr(
        main_module, "organize_and_form_input", fake_organize_and_form_input
    )


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
        "--output-dir",
        str(output_dir),
        "--model",
        "gpt-3.5-turbo-0125",
        "--model-temperature",
        "0.0",
        "--conv-round-limit",
        "15",
        "--num-processes",
        "1",
        "--task",
        "dummy_task",
        "--setup-map",
        str(setup_file),
        "--tasks-map",
        str(tasks_file),
    ]

    # Execute the main driver.
    main_module.main()

    # Assertions on fake function calls.
    assert (
        call_tracker["make_swe_tasks"] == 1
    ), "Expected make_swe_tasks to be called once."
    assert (
        call_tracker["group_swe_tasks_by_env"] == 1
    ), "Expected group_swe_tasks_by_env to be called once."

    # Check that the output file was created with expected content.
    swe_input_file = output_dir / "swe_input.txt"
    assert swe_input_file.exists(), "Expected the swe_input.txt file to be created."
    content = swe_input_file.read_text()
    assert (
        content == "dummy input content"
    ), "Output file content does not match expected content."


def test_github_issue(tmp_path):
    """
    Test the github-issue command:
      - Verify that the patched RawGithubTask constructor is called.
    """
    output_dir = tmp_path / "output"

    sys.argv = [
        "main.py",
        "github-issue",
        "--output-dir",
        str(output_dir),
        "--model",
        "gpt-3.5-turbo-0125",
        "--model-temperature",
        "0.0",
        "--conv-round-limit",
        "15",
        "--num-processes",
        "1",
        "--task-id",
        "dummy_task",
        "--clone-link",
        "https://example.com/dummy.git",
        "--commit-hash",
        "abc123",
        "--issue-link",
        "https://github.com/example/repo/issues/1",
        "--setup-dir",
        str(tmp_path / "setup_dir"),
    ]
    main_module.main()

    # Assert that the patched RawGithubTask constructor was invoked.
    assert (
        call_tracker["RawGithubTask"] == 1
    ), "Expected RawGithubTask to be instantiated once."


def test_local_issue(tmp_path):
    """
    Test the local-issue command:
      - Verify that the patched RawLocalTask constructor is called.
    """
    output_dir = tmp_path / "output"

    sys.argv = [
        "main.py",
        "local-issue",
        "--output-dir",
        str(output_dir),
        "--model",
        "gpt-3.5-turbo-0125",
        "--model-temperature",
        "0.0",
        "--conv-round-limit",
        "15",
        "--num-processes",
        "1",
        "--task-id",
        "dummy_task",
        "--local-repo",
        str(tmp_path / "dummy_repo"),
        "--issue-file",
        str(tmp_path / "dummy_issue.txt"),
    ]
    main_module.main()

    # Assert that the patched RawLocalTask constructor was invoked.
    assert (
        call_tracker["RawLocalTask"] == 1
    ), "Expected RawLocalTask to be instantiated once."


def test_run_raw_task_success(monkeypatch, tmp_path):
    """
    Test that run_raw_task returns True when do_inference returns True.
    It also verifies that dump_meta_data is called and that a dummy patch path is logged.
    """
    # Set the configuration so that we run the normal branch (not eval reproducer).
    config.only_eval_reproducer = False
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config.output_dir = str(output_dir)

    # Define a dummy task with the required methods.
    class DummyTask:
        task_id = "dummy123"

        def dump_meta_data(self, out_dir: str):
            # Simulate writing a meta file.
            Path(out_dir, "meta.json").write_text('{"dummy": "data"}')

        def to_task(self):
            return self

    dummy_task = DummyTask()

    # Patch do_inference to return True.
    monkeypatch.setattr(main_module, "do_inference", lambda task, out_dir: True)
    # Patch get_final_patch_path to return a dummy patch path.
    monkeypatch.setattr(
        main_module, "get_final_patch_path", lambda out_dir: "dummy_patch_path"
    )
    # Patch create_dir_if_not_exists to create the directory.
    monkeypatch.setattr(
        main_module.apputils,
        "create_dir_if_not_exists",
        lambda d: Path(d).mkdir(parents=True, exist_ok=True),
    )
    # Record log messages.
    log_messages = []
    monkeypatch.setattr(
        main_module.log, "log_and_always_print", lambda msg: log_messages.append(msg)
    )

    # Call the function under test.
    result = run_raw_task(dummy_task)

    # Assert that the dummy inference returned True.
    assert result is True
    # Check that dump_meta_data has created a meta.json in the task output directory.
    # We know the output directory is config.output_dir/dummy123_<timestamp>; we can check that some file exists.
    task_dirs = list(Path(config.output_dir).glob("dummy123_*"))
    assert (
        len(task_dirs) == 1
    ), "Expected one output directory to be created for the task."
    meta_file = task_dirs[0] / "meta.json"
    assert (
        meta_file.exists()
    ), "Expected meta.json to be dumped in the task output directory."
    # Check that a log message mentioning the dummy patch path was printed.
    assert any(
        "dummy_patch_path" in msg for msg in log_messages
    ), "Expected patch path log message."


def test_run_raw_task_exception(monkeypatch, tmp_path):
    """
    Test that run_raw_task catches exceptions from do_inference and returns False.
    It also logs an appropriate error message.
    """
    config.only_eval_reproducer = False
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config.output_dir = str(output_dir)

    class DummyTask:
        task_id = "dummy456"

        def dump_meta_data(self, out_dir: str):
            Path(out_dir, "meta.json").write_text('{"dummy": "data"}')

        def to_task(self):
            return self

    dummy_task = DummyTask()

    # Patch do_inference to raise an exception.
    def fake_inference(task, out_dir):
        raise ValueError("inference error")

    monkeypatch.setattr(main_module, "do_inference", fake_inference)
    # Patch get_final_patch_path to return None.
    monkeypatch.setattr(main_module, "get_final_patch_path", lambda out_dir: None)
    monkeypatch.setattr(
        main_module.apputils,
        "create_dir_if_not_exists",
        lambda d: Path(d).mkdir(parents=True, exist_ok=True),
    )
    # Record log messages.
    log_messages = []
    monkeypatch.setattr(
        main_module.log, "log_and_always_print", lambda msg: log_messages.append(msg)
    )
    # Patch logger.exception to do nothing.
    monkeypatch.setattr(main_module.logger, "exception", lambda e: None)

    result = run_raw_task(dummy_task)
    # Since an exception is raised inside do_inference, the function should return False.
    assert result is False
    # Check that a log message indicating failure was printed.
    assert any(
        "failed with exception" in msg for msg in log_messages
    ), "Expected error log message."


@pytest.mark.integration
def test_anthropic_api_integration():
    # Step 1: Use the key from the environment if provided; otherwise, fall back to a dummy key.
    dummy_key = "sk-ant-dummy"
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", dummy_key)
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key 
    # if anthropic_key == dummy_key:
    #     print("Using dummy ANTHROPIC_API_KEY:", anthropic_key)
    # else:
    #     print("Using ANTHROPIC_API_KEY:", "***")
    
    # Step 2: Construct the command to run.
    command = "conda run -n auto-code-rover env PYTHONPATH=$(pwd) python app/main.py github-issue --output-dir output --setup-dir setup --model claude-3-haiku-20240307 --model-temperature 0.2 --task-id langchain-20453 --clone-link https://github.com/langchain-ai/langchain.git --commit-hash cb6e5e5 --issue-link https://github.com/langchain-ai/langchain/issues/20453"
    print("Running command:", command)

    # Step 3: Run the command and capture stdout and stderr.
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Step 4: Log the captured output and error.
    print("=== Command Output ===")
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print("=== End Command Output ===\n")

    # Step 5: Assert based on expected outcome.
    # For a dummy key, we expect the API call to fail with an "Invalid API key" error message.
    expected_error_message = "AuthenticationError"
    if os.environ["ANTHROPIC_API_KEY"] == dummy_key:
        assert (
            expected_error_message in result.stderr
            or expected_error_message in result.stdout
        ), (
            f"Test failed: Expected error message '{expected_error_message}' not found.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    else:
        # For a real key scenario, you could assert a placeholder success message.
        expected_success_message = "Finished all tasks sequentially."
        assert (
            expected_success_message in result.stdout
        ), f"Test failed: Expected success message '{expected_success_message}' not found."


@pytest.mark.integration
def test_openai_simple():
    # Use the key from the environment if provided; otherwise, fall back to a dummy key.
    dummy_key = "sk-openai-dummy"
    openai_key = os.environ.get("OPENAI_KEY", dummy_key)
    openai.api_key = openai_key
    print("Using OPENAI_KEY:", openai_key)

    # If we're using a dummy key, skip the test.
    if openai_key == dummy_key:
        pytest.skip("No valid OPENAI_KEY provided; skipping live OpenAI API test.")

    try:
        # Call the OpenAI ChatCompletion endpoint.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, OpenAI!"}],
        )
    except Exception as e:
        pytest.fail(
            f"OpenAI API call failed: {e}\n"
            "If using openai>=1.0.0, run `openai migrate` or pin to an older version (e.g. openai==0.28)."
        )

    # Assert that the response contains choices.
    assert (
        "choices" in response and len(response["choices"]) > 0
    ), "No choices returned from OpenAI API."
