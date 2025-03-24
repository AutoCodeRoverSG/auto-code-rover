from tempfile import NamedTemporaryFile

from app.agents.patch_utils import *


def test_parse_edits():
    chat_input = (
        "```\n"
        "<file>test.py</file>\n"
        "<original>\n"
        "def foo():\n    pass\n"
        "</original>\n"
        "<patched>\n"
        "def foo():\n    print('patched')\n"
        "</patched>\n"
        "```"
    )

    edits = parse_edits(chat_input)
    assert len(edits) == 1
    assert edits[0].filename == "test.py"
    assert edits[0].before == "def foo():\n    pass"
    assert edits[0].after == "def foo():\n    print('patched')"


def test_apply_edit_success():
    original_content = "def foo():\n    pass\n"
    patched_content = "def foo():\n    print('patched')"

    edit = Edit("dummy.py", original_content.strip(), patched_content.strip())

    with NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(original_content)
        tmp_path = tmp.name

    result = apply_edit(edit, tmp_path)

    assert result == tmp_path

    with open(tmp_path) as f:
        assert f.read().strip() == patched_content.strip()


def test_apply_edit_failure():
    original_content = "def bar():\n    pass\n"
    edit = Edit("dummy.py", "def foo():\n    pass", "def foo():\n    print('patched')")

    with NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(original_content)
        tmp_path = tmp.name

    result = apply_edit(edit, tmp_path)

    assert result is None


def test_lint_python_content():
    valid_content = "def foo():\n    return True"
    invalid_content = "def foo(\n    return True"

    assert lint_python_content(valid_content) is True
    assert lint_python_content(invalid_content) is False
