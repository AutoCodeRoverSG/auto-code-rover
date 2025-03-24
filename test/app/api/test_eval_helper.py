import json
import re
import textwrap
from enum import Enum
from pathlib import Path

from app.api import eval_helper
from app.api.eval_helper import *


# Prevent pytest from collecting the helper functions that start with "test_" but are not actually unit tests
# (they are intended to be used as helpers in other modules).
eval_helper.test_passed.__test__ = False
eval_helper.test_failed.__test__ = False


# --- Tests for the log parsers ---


def test_parse_log_pytest():
    # Sample log lines for pytest format.
    # e.g. "PASSED test_func1" and "FAILED test_func2 - some error"
    log = textwrap.dedent(
        """\
        PASSED test_func1
        FAILED test_func2 - AssertionError
        SKIPPED test_func3
        ERROR test_func4 - Exception
    """
    )
    result = parse_log_pytest(log)
    # We expect mapping: test_func1 -> PASSED, test_func2 -> FAILED, etc.
    assert result.get("test_func1") == "PASSED"
    assert result.get("test_func2") == "FAILED"
    assert result.get("test_func3") == "SKIPPED"
    assert result.get("test_func4") == "ERROR"


def test_parse_log_django():
    # Django logs typically have patterns like:
    # "some_test ... ok", "another_test ... skipped", "yetanother_test ... FAIL", etc.
    log = textwrap.dedent(
        """\
        test_app.tests.TestSomething.test_one ... ok
        test_app.tests.TestSomething.test_two ... skipped
        test_app.tests.TestSomething.test_three ... FAIL
        FAIL: test_app.tests.TestSomething.test_four
        ERROR: test_app.tests.TestSomething.test_five
        test_app.tests.TestSomething.test_six ... ERROR
    """
    )
    result = parse_log_django(log)
    # Expected status based on our parser.
    assert (
        result.get("test_app.tests.TestSomething.test_one") == TestStatus.PASSED.value
    )
    assert (
        result.get("test_app.tests.TestSomething.test_two") == TestStatus.SKIPPED.value
    )
    assert (
        result.get("test_app.tests.TestSomething.test_three") == TestStatus.FAILED.value
    )
    assert (
        result.get("test_app.tests.TestSomething.test_four") == TestStatus.FAILED.value
    )
    assert (
        result.get("test_app.tests.TestSomething.test_five") == TestStatus.ERROR.value
    )
    assert result.get("test_app.tests.TestSomething.test_six") == TestStatus.ERROR.value


def test_parse_log_pytest_v2():
    # Sample log for pytest v2: includes ANSI escape sequences and a hunk for FAILED.
    log = "\x1b[31mFAILED\x1b[0m test_func_v2 - error message"
    result = parse_log_pytest_v2(log)
    # The escape sequences should be removed; we expect test_func_v2 mapped to FAILED.
    assert result.get("test_func_v2") == "FAILED"


def test_parse_log_seaborn():
    # Seaborn log sample: failed line starts with "FAILED", passed line has " PASSED " in it.
    log = textwrap.dedent(
        """\
        dummy_test PASSED some extra text
        FAILED another_test
    """
    )
    result = parse_log_seaborn(log)
    # For FAILED, we split and take the second token.
    assert result.get("another_test") == TestStatus.FAILED.value
    # For PASSED, if the second token equals PASSED, then key is the first token.
    assert result.get("dummy_test") == TestStatus.PASSED.value


def test_parse_log_sympy():
    # Sample sympy log: first part uses regex and then additional lines.
    # Create a fake match pattern. The regex pattern in parse_log_sympy is:
    # r"(_*) (.*)\.py:(.*) (_*)"
    # We can simulate one match and then additional lines.
    log = textwrap.dedent(
        """\
        ____ dummy.py:10 ____
        test_sympy1 E
        test_sympy2 F
        test_sympy3 ok
    """
    )
    result = parse_log_sympy(log)
    # From regex part, we expect one entry for "dummy.py:10"
    assert "dummy.py:10" in result
    # And additional lines produce mappings:
    assert result.get("test_sympy1") == TestStatus.ERROR.value
    assert result.get("test_sympy2") == TestStatus.FAILED.value
    assert result.get("test_sympy3") == TestStatus.PASSED.value


# --- Tests for get_logs_eval ---


def test_get_logs_eval_success(tmp_path):
    # Create a temporary log file with a valid log (using pytest parser).
    log_content = "PASSED test_eval1\nFAILED test_eval2"
    log_file = tmp_path / "log.txt"
    log_file.write_text(log_content)
    # Use a repo that maps to parse_log_pytest, e.g. "pytest-dev/pytest"
    parsed, ok = get_logs_eval("pytest-dev/pytest", str(log_file))
    assert ok is True
    assert parsed.get("test_eval1") == "PASSED"
    assert parsed.get("test_eval2") == "FAILED"


def test_get_logs_eval_failure(tmp_path):
    # Create a temporary log file with error markers.
    log_content = f"{TESTS_ERROR}\nSome error occurred."
    log_file = tmp_path / "log_error.txt"
    log_file.write_text(log_content)
    parsed, ok = get_logs_eval("pytest-dev/pytest", str(log_file))
    # In case of error, we expect an empty dict and ok False.
    assert ok is False
    assert parsed == {}


# --- Tests for test_passed and test_failed ---


def test_test_passed_and_failed():
    # Create a simple status mapping.
    status_map = {
        "case1": TestStatus.PASSED.value,
        "case2": TestStatus.FAILED.value,
        "case3": TestStatus.ERROR.value,
    }
    # test_passed returns True if test exists and status is PASSED.
    assert test_passed("case1", status_map) is True
    # For a case that is FAILED or ERROR, test_passed should be False.
    assert test_passed("case2", status_map) is False
    # test_failed returns True if test is not present or is FAILED/ERROR.
    assert test_failed("case2", status_map) is True
    # If a test is PASSED, test_failed should be False.
    assert test_failed("case1", status_map) is False


# --- Tests for get_eval_report, compute_fail_to_pass, compute_pass_to_pass, get_resolution_status ---


def test_get_eval_report():
    # Create dummy gold results.
    gold = {
        FAIL_TO_PASS: ["t1", "t2"],
        PASS_TO_PASS: ["t3", "t4"],
        FAIL_TO_FAIL: ["t5"],
        PASS_TO_FAIL: ["t6"],
    }
    # Create an evaluation status map.
    eval_sm = {
        "t1": TestStatus.PASSED.value,  # success for FAIL_TO_PASS
        "t2": TestStatus.FAILED.value,  # failure for FAIL_TO_PASS
        "t3": TestStatus.PASSED.value,  # success for PASS_TO_PASS
        "t4": TestStatus.FAILED.value,  # failure for PASS_TO_PASS
        "t5": TestStatus.PASSED.value,  # for extra credit (if calculated)
        "t6": TestStatus.FAILED.value,  # not considered if calculated
    }
    report = get_eval_report(eval_sm, gold, calculate_to_fail=True)
    # For FAIL_TO_PASS, success count should be 1 and failure count 1.
    assert report[FAIL_TO_PASS]["success"] == ["t1"]
    assert report[FAIL_TO_PASS]["failure"] == ["t2"]
    # For PASS_TO_PASS, success count 1 and failure count 1.
    assert report[PASS_TO_PASS]["success"] == ["t3"]
    assert report[PASS_TO_PASS]["failure"] == ["t4"]
    # And extra metrics for FAIL_TO_FAIL and PASS_TO_FAIL.
    assert report[FAIL_TO_FAIL]["success"] == ["t5"]
    assert report[PASS_TO_FAIL]["failure"] == ["t6"]


def test_compute_metrics_and_resolution():
    # Create a sample report.
    report = {
        FAIL_TO_PASS: {"success": ["t1", "t2"], "failure": ["t3"]},
        PASS_TO_PASS: {"success": ["t4"], "failure": ["t5"]},
    }
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)
    # f2p should be 2/(2+1) = 0.666..., p2p should be 1/(1+1)=0.5.
    assert abs(f2p - 0.6666) < 0.01
    assert abs(p2p - 0.5) < 0.01

    # Test get_resolution_status:
    # Case FULL: f2p==1 and p2p==1.
    report_full = {
        FAIL_TO_PASS: {"success": ["t1"], "failure": []},
        PASS_TO_PASS: {"success": ["t2"], "failure": []},
    }
    status_full = get_resolution_status(report_full)
    assert status_full == ResolvedStatus.FULL

    # Case PARTIAL: f2p between 0 and 1, p2p==1.
    report_partial = {
        FAIL_TO_PASS: {"success": ["t1"], "failure": ["t3"]},
        PASS_TO_PASS: {"success": ["t2"], "failure": []},
    }
    status_partial = get_resolution_status(report_partial)
    assert status_partial == ResolvedStatus.PARTIAL

    # Otherwise, status NO.
    report_no = {
        FAIL_TO_PASS: {"success": [], "failure": ["t1"]},
        PASS_TO_PASS: {"success": ["t2"], "failure": []},
    }
    status_no = get_resolution_status(report_no)
    assert status_no == ResolvedStatus.NO
