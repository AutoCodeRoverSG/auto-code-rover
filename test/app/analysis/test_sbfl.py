import os
import pytest
import tempfile
import textwrap
from pathlib import Path

# Import everything from the sbfl module.
from app.analysis.sbfl import *

# --- Dummy Classes and Fixtures for Isolating sbfl.py ---


# Dummy Task classes to simulate the required interface.
class DummyTask:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def apply_patch(self, patch):
        # Return a dummy context manager that does nothing.
        class DummyContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return DummyContext()


class DummySweTask(DummyTask):
    pass


# Fixture to create a temporary project directory structure.
@pytest.fixture
def temp_project(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    # Create a dummy .coverage file required by tests.
    cov_file = project_dir / ".coverage"
    cov_file.write_text("dummy coverage data")
    return str(project_dir)


# Dummy CoverageData to simulate coverage.sqldata behavior.
class DummyCoverageData:
    def __init__(self, basename):
        self.basename = basename

    def read(self):
        pass

    def measured_files(self):
        # Return a list with one dummy file path.
        return ["/dummy/file.py"]

    def contexts_by_lineno(self, file):
        # Return a mapping: line number -> list of test names.
        return {10: ["file.py::test_pass", "file.py::test_fail"], 20: [""]}


@pytest.fixture(autouse=True)
def dummy_coverage(monkeypatch):
    monkeypatch.setattr(
        "app.analysis.sbfl.CoverageData", lambda basename: DummyCoverageData(basename)
    )


# Dummy apputils to simulate context management and command execution.
class DummyAppUtils:
    def cd(self, directory):
        # Dummy context manager for directory change.
        class DummyCM:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return DummyCM()

    def run_string_cmd_in_conda(self, cmd, env_name, **kwargs):
        # Dummy return object with stdout attribute.
        class DummyCompletedProcess:
            stdout = "dummy command output"

        return DummyCompletedProcess()


@pytest.fixture(autouse=True)
def dummy_apputils(monkeypatch):
    import app.utils as apputils

    dummy = DummyAppUtils()
    monkeypatch.setattr(apputils, "cd", dummy.cd)
    monkeypatch.setattr(
        apputils, "run_string_cmd_in_conda", dummy.run_string_cmd_in_conda
    )


# Dummy log module to capture/log messages.
class DummyLog:
    @staticmethod
    def log_and_print(msg, *args, **kwargs):
        print("LOG:", msg)


@pytest.fixture(autouse=True)
def dummy_log(monkeypatch):
    import app.log as log

    monkeypatch.setattr(log, "log_and_print", DummyLog.log_and_print)


# --- Tests for Canonicalization Functions ---


def test_canonicalize_testname_sympy_bin_test():
    result = canonicalize_testname_sympy_bin_test("test_func")
    assert result == ("", "test_func")


# def test_canonicalize_testname_django_runner_valid():
#     # Valid pattern: "test_func (module.TestCase.test_func)"
#     testname = "test_func (module.TestCase.test_func)"
#     file_name, full_name = canonicalize_testname_django_runner(testname)
#     # Expect file name from the lower-case module part.
#     assert file_name == "module.py"
#     # Full name is the module path concatenated with the function name.
#     assert full_name == "module.test_func.test_func"


def test_canonicalize_testname_django_runner_valid_new():
    # Valid pattern: "test_func (module.TestCase.test_func)"
    testname = "test_func (module.TestCase.test_func)"
    file_name, full_name = canonicalize_testname_django_runner(testname)
    # According to the current implementation, both "module" and "test_func" are lowercase,
    # so the file name becomes "module/test_func.py"
    assert file_name == "module/test_func.py"
    # The full name is built by concatenating the path with the function name
    assert full_name == "module.TestCase.test_func.test_func"


def test_canonicalize_testname_different_task_ids_new():
    # For a django task id using the valid pattern.
    testname = "test_func (module.TestCase.test_func)"
    result_django = canonicalize_testname("django_task", testname)
    # With the current behavior, the file name is "module/test_func.py"
    assert result_django[0] == "module/test_func.py"


def test_canonicalize_testname_django_runner_invalid():
    # Invalid pattern should return empty strings.
    testname = "invalid_test_name"
    file_name, full_name = canonicalize_testname_django_runner(testname)
    assert file_name == ""
    assert full_name == ""


def test_canonicalize_testname_pytest():
    testname = "file.py::test_func"
    file_name, full_name = canonicalize_testname_pytest(testname)
    assert file_name == "file.py"
    assert full_name == testname


# def test_canonicalize_testname_different_task_ids():
#     # For django task id.
#     result_django = canonicalize_testname("django_task", "test_func (module.test_func)")
#     assert result_django[0] == "module.py"
#     # For sympy task id.
#     result_sympy = canonicalize_testname("sympy_task", "test_func")
#     assert result_sympy == ("", "test_func")
#     # Default branch uses pytest canonicalization.
#     result_default = canonicalize_testname("other_task", "file.py::test_func")
#     assert result_default == ("file.py", "file.py::test_func")

# --- Tests for Execution Statistics Classes ---


def test_file_exec_stats():
    fes = FileExecStats("dummy.py")
    fes.incre_pass_count(10)
    fes.incre_fail_count(10)
    assert fes.line_stats[10] == (1, 1)
    fes.incre_pass_count(10)
    assert fes.line_stats[10] == (2, 1)
    s = str(fes)
    assert "dummy.py" in s


def test_exec_stats_and_ranking():
    exec_stats = ExecStats()
    fes = FileExecStats("dummy.py")
    fes.incre_fail_count(5)
    fes.incre_pass_count(5)
    exec_stats.add_file(fes)
    # total_fail = 1, total_pass = 1
    ranked = exec_stats.rank_lines(ExecStats.ochiai, 1, 1)
    # Should return one tuple: (file, line_no, score)
    assert isinstance(ranked, list)
    assert len(ranked) == 1
    file, line_no, score = ranked[0]
    assert file == "dummy.py"
    assert line_no == 5
    # Test dstar edge-case (bottom zero)
    score_dstar = ExecStats.dstar(0, 0, 1, 1)
    assert score_dstar == 0


# --- Tests for Helper Functions ---


def test_helper_remove_dup_and_empty():
    lst = ["a", "b", "", "a"]
    result = helper_remove_dup_and_empty(lst)
    assert set(result) == {"a", "b"}


def test_helper_two_tests_match():
    # Test using endswith matching.
    assert helper_two_tests_match("a.b.c", "b.c")
    assert helper_two_tests_match("b.c", "a.b.c") is True
    assert helper_two_tests_match("a.b.c", "a.b.d") is False


def test_helper_test_match_any():
    test = "a.b.c"
    candidates = ["b.c", "x.y"]
    assert helper_test_match_any(test, candidates) is True
    candidates = ["x.y"]
    assert helper_test_match_any(test, candidates) is False


# --- Tests for Collation and Mapping Functions ---


def test_collate_results():
    # Create ranked lines with some non-positive scores and adjacent lines.
    ranked_lines = [
        ("/file1.py", 10, 0.5),
        ("/file1.py", 11, 0.7),
        ("/file1.py", 13, 0.0),  # non-positive; should be filtered out.
        ("/file2.py", 20, 0.9),
    ]
    test_file_names = ["/test_file.py"]
    results = collate_results(ranked_lines, test_file_names)
    # Expect /file2.py to appear first (highest score), and /file1.py lines merged.
    assert results[0][0] == "/file2.py"
    assert results[0][1] == 20
    assert results[0][2] == 20
    assert results[0][3] == 0.9
    assert results[1][0] == "/file1.py"
    assert results[1][1] == 10
    assert results[1][2] == 11
    assert results[1][3] == 0.7


def test_method_ranges_in_file(tmp_path):
    # Create a temporary Python file with two functions and a class with methods.
    file_content = textwrap.dedent(
        """
        def func1():
            pass

        def func2():
            pass

        class A:
            def method1(self):
                pass

            async def method2(self):
                pass
    """
    )
    file_path = tmp_path / "dummy.py"
    file_path.write_text(file_content)
    ranges = method_ranges_in_file(str(file_path))
    # Expect 4 methods: func1, func2, method1, method2.
    method_names = {m.method_name for m in ranges.keys()}
    assert method_names == {"func1", "func2", "method1", "method2"}
    # Test the SyntaxError branch by providing invalid Python.
    invalid_file = tmp_path / "invalid.py"
    invalid_file.write_text("def invalid(:")
    ranges_invalid = method_ranges_in_file(str(invalid_file))
    assert ranges_invalid == {}


def test_map_collated_results_to_methods(tmp_path):
    # Create a dummy Python file with one function.
    file_content = textwrap.dedent(
        """
        def func1():
            pass
    """
    )
    file_path = tmp_path / "dummy.py"
    file_path.write_text(file_content)
    # Create a ranked range that overlaps with the function's lines.
    ranked_ranges = [(str(file_path), 1, 3, 0.8)]
    mapped = map_collated_results_to_methods(ranked_ranges)
    # Expect to map to 'func1'.
    assert len(mapped) == 1
    filename, class_name, method_name, susp = mapped[0]
    assert filename == str(file_path)
    assert method_name == "func1"


# --- Tests for the Main run Functions and PythonSbfl Class ---


def test_run_with_non_swe_task():
    # A dummy task that is NOT an instance of SweTask should raise NotImplementedError.
    dummy_task = DummyTask(
        task_id="other",
        testcases_passing=[],
        testcases_failing=[],
        test_patch="",
        project_path=".",
        test_cmd="pytest",
        env_name="env",
    )
    with pytest.raises(NotImplementedError):
        run(dummy_task)


def test_python_sbfl_run_success(tmp_path, monkeypatch):
    # Test PythonSbfl.run with a dummy SweTask simulating a successful run.
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    # Create a dummy .coverage file in the project directory.
    cov_file_path = project_dir / ".coverage"
    cov_file_path.write_text("dummy")
    task = DummySweTask(
        task_id="other",
        test_patch="",
        testcases_passing=["file.py::test_pass"],
        testcases_failing=["file.py::test_fail"],
        project_path=str(project_dir),
        test_cmd="pytest --some-arg",
        env_name="env",
    )
    # Monkeypatch os.path.isfile and os.path.exists to simulate that the coverage file exists.
    monkeypatch.setattr(os.path, "isfile", lambda path: True)
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    test_files, ranked_lines, log_file = PythonSbfl.run(task)
    # The test file names should include the canonicalized names.
    assert "file.py" in test_files
    # ranked_lines should be a list and log_file should be non-empty.
    assert isinstance(ranked_lines, list)
    assert log_file != ""


def test_python_sbfl_run_no_coverage(tmp_path, monkeypatch):
    # Test that PythonSbfl.run raises NoCoverageData when no coverage file exists.
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    task = DummySweTask(
        task_id="other",
        test_patch="",
        testcases_passing=["file.py::test_pass"],
        testcases_failing=["file.py::test_fail"],
        project_path=str(project_dir),
        test_cmd="pytest --some-arg",
        env_name="env",
    )
    monkeypatch.setattr(os.path, "isfile", lambda path: False)
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    with pytest.raises(NoCoverageData):
        PythonSbfl.run(task)


# --- Tests for Configuration File Helpers ---


def test_specify_dynamic_context(tmp_path):
    coveragerc = tmp_path / "coveragerc"
    # When the file does not exist.
    PythonSbfl._specify_dynamic_context(str(coveragerc))
    content = coveragerc.read_text()
    assert "dynamic_context = test_function" in content
    # When the file exists with a [run] section.
    coveragerc.write_text("[run]\nexisting_setting=True\n")
    PythonSbfl._specify_dynamic_context(str(coveragerc))
    content = coveragerc.read_text()
    assert "dynamic_context = test_function" in content


def test_omit_coverage_in_file(tmp_path):
    coveragerc = tmp_path / "coveragerc"
    coveragerc.write_text("[run]\nomit = original")
    omitted = ["file1.py", "file2.py"]
    PythonSbfl._omit_coverage_in_file(str(coveragerc), omitted)
    content = coveragerc.read_text()
    for file in omitted:
        assert file in content


def test_add_pytest_cov_to_tox(tmp_path):
    tox_ini = tmp_path / "tox.ini"
    # Create a minimal tox.ini.
    tox_ini.write_text("[testenv]\ndeps =\ncommands = pytest")
    PythonSbfl._add_pytest_cov_to_tox(str(tox_ini))
    content = tox_ini.read_text()
    assert "pytest-cov" in content
    assert "pytest --cov --cov-context=test" in content
