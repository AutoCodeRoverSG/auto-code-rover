import re
import time
from io import StringIO
import pytest

from app.api.log import (
    terminal_width,
    WIDTH,
    console,
    print_stdout,
    log_exception,
    print_banner,
    replace_html_tags,
    print_acr,
    print_retrieval,
    print_patch_generation,
    print_issue,
    print_reproducer,
    print_exec_reproducer,
    print_review,
    log_and_print,
    log_and_cprint,
    log_and_always_print,
    print_with_time,
)
from loguru import logger
from rich.panel import Panel

# A dummy console to record print calls.
class DummyConsole:
    def __init__(self):
        self.calls = []
    def print(self, *args, **kwargs):
        self.calls.append((args, kwargs))

# Automatically restore the module-level print_stdout after each test.
@pytest.fixture(autouse=True)
def restore_print_stdout(monkeypatch):
    original = print_stdout
    monkeypatch.setattr("log.print_stdout", True)
    yield
    monkeypatch.setattr("log.print_stdout", original)

def test_replace_html_tags():
    input_str = "<file>code</file> and <class>MyClass</class>"
    expected = "[file]code[/file] and [class]MyClass[/class]"
    assert replace_html_tags(input_str) == expected

def test_print_banner(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)

    msg = "Test Banner"
    print_banner(msg)
    # print_banner prints an empty line, then the banner, then another empty line.
    assert len(dummy.calls) == 3
    banner = f" {msg} ".center(WIDTH, "=")
    # The second call should print the banner with style "bold"
    args, kwargs = dummy.calls[1]
    assert args[0] == banner
    assert kwargs.get("style") == "bold"

def test_print_acr(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)

    msg = "<file>Test</file>"
    desc = "desc"
    print_acr(msg, desc)
    found = False
    # Look for a Panel with the expected title.
    for args, kwargs in dummy.calls:
        for arg in args:
            if isinstance(arg, Panel) and arg.title == f"AutoCodeRover ({desc})":
                found = True
    assert found

def test_print_retrieval(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)

    msg = "<class>Example</class>"
    desc = "retrieval"
    print_retrieval(msg, desc)
    found = False
    for args, kwargs in dummy.calls:
        for arg in args:
            if isinstance(arg, Panel) and arg.title == f"Context Retrieval Agent ({desc})":
                found = True
    assert found

def test_print_patch_generation(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)

    msg = "<func>patch</func>"
    desc = "patch"
    print_patch_generation(msg, desc)
    found = False
    for args, kwargs in dummy.calls:
        for arg in args:
            if isinstance(arg, Panel) and arg.title == f"Patch Generation ({desc})":
                found = True
    assert found

def test_print_issue(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)

    content = "Issue content"
    print_issue(content)
    found = False
    for args, kwargs in dummy.calls:
        for arg in args:
            if isinstance(arg, Panel) and arg.title == "Issue description":
                found = True
    assert found

def test_print_reproducer(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)

    msg = "Reproducer message"
    desc = "reproducer"
    print_reproducer(msg, desc)
    found = False
    for args, kwargs in dummy.calls:
        for arg in args:
            if isinstance(arg, Panel) and arg.title == f"Reproducer Test Generation ({desc})":
                found = True
    assert found

def test_print_exec_reproducer(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)

    msg = "Execution message"
    desc = "exec"
    print_exec_reproducer(msg, desc)
    found = False
    for args, kwargs in dummy.calls:
        for arg in args:
            if isinstance(arg, Panel) and arg.title == f"Reproducer Execution Result ({desc})":
                found = True
    assert found

def test_print_review(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)

    msg = "Review message"
    desc = "review"
    print_review(msg, desc)
    found = False
    for args, kwargs in dummy.calls:
        for arg in args:
            if isinstance(arg, Panel) and arg.title == f"Review ({desc})":
                found = True
    assert found

def test_log_exception(monkeypatch):
    log_calls = []
    monkeypatch.setattr(logger, "exception", lambda exc: log_calls.append(str(exc)))
    exc = Exception("Test exception")
    log_exception(exc)
    assert any("Test exception" in call for call in log_calls)

def test_log_and_print(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)
    log_calls = []
    monkeypatch.setattr(logger, "info", lambda msg: log_calls.append(msg))

    msg = "Logging info"
    log_and_print(msg)
    assert msg in log_calls
    found = any(msg == args[0] for args, kwargs in dummy.calls)
    assert found

def test_log_and_cprint(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)
    log_calls = []
    monkeypatch.setattr(logger, "info", lambda msg: log_calls.append(msg))

    msg = "Logging cprint"
    log_and_cprint(msg, style="underline")
    assert msg in log_calls
    found = any(msg == args[0] for args, kwargs in dummy.calls)
    style_found = any(kwargs.get("style") == "underline" for args, kwargs in dummy.calls)
    assert found and style_found

def test_log_and_always_print(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)
    log_calls = []
    monkeypatch.setattr(logger, "info", lambda msg: log_calls.append(msg))

    msg = "Always print message"
    log_and_always_print(msg)
    assert msg in log_calls
    printed = False
    for args, kwargs in dummy.calls:
        for arg in args:
            if msg in str(arg) and re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", str(arg)):
                printed = True
    assert printed

def test_print_with_time(monkeypatch):
    dummy = DummyConsole()
    monkeypatch.setattr(console, "print", dummy.print)

    msg = "Message with time"
    print_with_time(msg)
    printed = False
    for args, kwargs in dummy.calls:
        for arg in args:
            if msg in str(arg) and re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", str(arg)):
                printed = True
    assert printed
