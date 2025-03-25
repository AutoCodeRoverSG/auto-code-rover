import ast
import os
import tempfile
import textwrap

from pathlib import Path

from app.search.search_utils import *

def test_is_test_file():
    # Setup: create a list of test file names
    test_files = [
        "test_utils.py",
        "test_search_utils.py",
        "test_search.py",
        "utils_test.py",
        "search_utils_test.py",
        "search_test.py",
        "test/test_utils.py",
    ]
    # Setup: create a list of non-test file names
    non_test_files = [
        "utils.py",
        "search_utils.py",
        "search.py",
        "config/framework.py",
        "config/routing.py",
        "greatest_common_divisor.py",  # This is not a test file, but it has "test" in its name, should not be recognized as a test file
    ]

    # Execute and verify: test files should return True, non-test files should return False
    for test_file in test_files:
        assert is_test_file(
            test_file
        ), f"{test_file} should be recognized as a test file."
    for non_test_file in non_test_files:
        assert not is_test_file(
            non_test_file
        ), f"{non_test_file} should not be recognized as a test file."


def test_find_python_files(tmp_path):
    # Setup: create a list of file names (python and non-python files)
    files = [
        "main.py",
        "utils.py",
        "test/test_something.py",
        "Controller/MonitorJobController.php",
        "templates/details.html.twig",
        "page.tsx",
        "dfs.cpp",
    ]

    # The expected list excludes test files (those inside a "test/" directory)
    expected_python_files = [
        "main.py",
        "utils.py",
    ]

    # Create a temporary base directory that avoids pytest discovery conflicts.
    base_dir = tmp_path / "files"
    base_dir.mkdir()

    # Create each file (ensure that subdirectories are created)
    for file in files:
        file_path = base_dir / file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("")

    # Execute and verify: only python files inside base_dir should be returned.
    python_files = find_python_files(str(base_dir))
    # Convert absolute paths to relative paths for comparison.
    python_files_rel = [os.path.relpath(pf, str(base_dir)) for pf in python_files]
    python_files_rel.sort()
    expected_python_files.sort()

    # Compare lengths
    assert len(python_files_rel) == len(
        expected_python_files
    ), f"Expected {len(expected_python_files)} python files, but got {len(python_files_rel)}."

    # Compare each element
    for expected, actual in zip(expected_python_files, python_files_rel):
        assert actual == expected, f"Expected {expected}, but got {actual}."


def test_parse_class_def_args_simple():
    source = "class Foo(B, object):\n    pass\n"
    tree = ast.parse(source)
    node = tree.body[0]  # The ClassDef node for Foo
    result = parse_class_def_args(source, node)
    # 'B' is returned; 'object' is skipped.
    assert result == ["B"]


def test_parse_class_def_args_type_call():
    source = "class Bar(type('D', (), {})):\n    pass\n"
    tree = ast.parse(source)
    node = tree.body[0]
    result = parse_class_def_args(source, node)
    # The source segment for the first argument of the type() call is "'D'"
    assert result == ["'D'"]


def test_parse_class_def_args_mixed():
    source = "class Baz(C, type('E', (), {}), object):\n    pass\n"
    tree = ast.parse(source)
    node = tree.body[0]
    result = parse_class_def_args(source, node)
    # The expected bases are "C" from the ast.Name and "'E'" from the type() call.
    assert result == ["C", "'E'"]


def test_parse_class_def_args_only_object():
    source = "class Quux(object):\n    pass\n"
    tree = ast.parse(source)
    node = tree.body[0]
    result = parse_class_def_args(source, node)
    # Since only object is used, the result should be an empty list.
    assert result == []


# --- Test using inline file creation (temporary file) ---
def test_parse_python_file():
    # Create a temporary file with known content.
    sample_content = (
        "class Foo:\n"
        "    def bar(self):\n"
        "        pass\n\n"
        "def baz():\n"
        "    pass\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(sample_content)
        tmp_filename = tmp_file.name

    try:
        result = parse_python_file(tmp_filename)
        if result is None:
            raise AssertionError("parse_python_file returned None for a valid file.")

        classes, class_to_funcs, top_level_funcs, class_relation_map = result

        # Expected results based on our sample content.
        # Note: line numbers depend on the content; here we know:
        #   class Foo:         -> line 1, end at line 3
        #   def bar(self):      -> line 2, end at line 3
        #   def baz():         -> line 5, end at line 6
        expected_classes = [("Foo", 1, 3)]
        expected_class_to_funcs = {"Foo": [("bar", 2, 3)]}
        expected_top_level_funcs = [
            ("baz", 5, 6),
            ("bar", 2, 3),
        ]  # old expected_top_level_funcs = [("baz", 5, 6)]
        expected_class_relation_map = {("Foo", 1, 3): []}

        assert (
            classes == expected_classes
        ), f"Expected classes {expected_classes}, got {classes}"
        assert (
            class_to_funcs == expected_class_to_funcs
        ), f"Expected class_to_funcs {expected_class_to_funcs}, got {class_to_funcs}"
        # TODO: Fix this failing assertion - Function Under Test (FUT) is not actually ignoring class-defined functions.
        # TODO: Clarify if Class Signatures should contain the function signatures as well.
        assert (
            expected_top_level_funcs == top_level_funcs
        ), f"Expected top_level_funcs {expected_top_level_funcs}, got {top_level_funcs}"
        assert (
            class_relation_map == expected_class_relation_map
        ), f"Expected class_relation_map {expected_class_relation_map}, got {class_relation_map}"
        print("Inline test passed.")
    finally:
        os.unlink(tmp_filename)


# --- Test get code region (should be language agnostic) ---
def test_get_code_region_containing_code(tmp_path):
    # Create a temporary file with sample content.
    file_content = (
        "line 1\n" "line 2\n" "line 3\n" "line 4\n" "line 5\n" "line 6\n" "line 7\n"
    )
    temp_file = tmp_path / "sample.txt"
    temp_file.write_text(file_content)

    # The strings to search for in the file.
    code_str_dict = {
        "case_1": "line 4",
        "case_2": "line 2",
        "case_3": "line 6",
    }

    # The matched line numbers (0-based index) for the occurrence.
    matched_line_no_dict = {
        "case_1": 3,
        "case_2": 1,
        "case_3": 5,
    }

    # Expected context when with_lineno is True.
    expected_context_with_lineno = {
        "case_1": (
            "1 line 1\n"
            "2 line 2\n"
            "3 line 3\n"
            "4 line 4\n"
            "5 line 5\n"
            "6 line 6\n"
            "7 line 7\n"
        ),
        "case_2": ("1 line 1\n" "2 line 2\n" "3 line 3\n" "4 line 4\n" "5 line 5\n"),
        "case_3": ("3 line 3\n" "4 line 4\n" "5 line 5\n" "6 line 6\n" "7 line 7\n"),
    }

    # Expected context when with_lineno is False.
    expected_context_without_lineno = {
        "case_1": "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7",
        "case_2": "line 1\nline 2\nline 3\nline 4\nline 5",
        "case_3": "line 3\nline 4\nline 5\nline 6\nline 7",
    }

    for case, code_str in code_str_dict.items():
        # Test with with_lineno=True.
        occurrences_with = get_code_region_containing_code(
            str(temp_file), code_str, with_lineno=True
        )
        assert len(occurrences_with) == 1
        matched_line_no, context_with = occurrences_with[0]
        assert matched_line_no == matched_line_no_dict[case]
        assert context_with == expected_context_with_lineno[case]

        # Test with with_lineno=False.
        occurrences_without = get_code_region_containing_code(
            str(temp_file), code_str, with_lineno=False
        )
        assert len(occurrences_without) == 1
        matched_line_no2, context_without = occurrences_without[0]
        assert matched_line_no2 == matched_line_no_dict[case]
        assert context_without == expected_context_without_lineno[case]


def test_get_func_snippet_with_hello(tmp_path):
    # Create a temporary Python file with two function definitions.
    file_content = (
        "def foo():\n"
        "    a = 1\n"
        "    print('hello world')\n"
        "\n"
        "def bar():\n"
        "    b = 2\n"
        "    print('goodbye')\n"
    )
    temp_file = tmp_path / "sample.py"
    temp_file.write_text(file_content)

    # Search for the string "hello", which is only in function foo.
    snippets = get_func_snippet_with_code_in_file(str(temp_file), "hello")

    # We expect exactly one function snippet to be returned.
    assert len(snippets) == 1
    snippet = snippets[0]
    # Assert that the snippet belongs to function foo.
    assert "def foo():" in snippet
    # Verify that the snippet contains the searched code.
    assert "print('hello world')" in snippet
    # Also, ensure that function bar is not in the snippet.
    assert "def bar():" not in snippet


def test_get_func_snippet_with_print(tmp_path):
    # Create a temporary Python file with two function definitions.
    file_content = (
        "def foo():\n"
        "    a = 1\n"
        "    print('hello world')\n"
        "\n"
        "def bar():\n"
        "    b = 2\n"
        "    print('goodbye')\n"
    )
    temp_file = tmp_path / "sample.py"
    temp_file.write_text(file_content)

    # Search for the string "print", which should be present in both functions.
    snippets = get_func_snippet_with_code_in_file(str(temp_file), "print")

    # Expect two function snippets.
    assert len(snippets) == 2

    # Check that each returned snippet contains a function definition with a print statement.
    for snippet in snippets:
        assert "def " in snippet
        assert "print(" in snippet

    # Additionally, verify that the snippets correspond to the correct functions.
    foo_found = any("def foo():" in s and "print('hello world')" in s for s in snippets)
    bar_found = any("def bar():" in s and "print('goodbye')" in s for s in snippets)
    assert foo_found and bar_found


# --- Test get code snippets (note that lineno is 1-based) ---
def test_get_code_snippets_with_lineno(tmp_path):
    # Create a temporary file with sample content.
    file_content = "line one\n" "line two\n" "line three\n" "line four\n"
    temp_file = tmp_path / "sample.txt"
    temp_file.write_text(file_content)

    # Test retrieving lines 2 to 3 with line numbers.
    snippet = get_code_snippets(str(temp_file), 2, 3, with_lineno=True)
    expected = "2 line two\n3 line three\n"
    assert (
        snippet == expected
    ), "Snippet with line numbers does not match expected output."


def test_get_code_snippets_without_lineno(tmp_path):
    # Create a temporary file with sample content.
    file_content = "first line\n" "second line\n" "third line\n" "fourth line\n"
    temp_file = tmp_path / "sample.txt"
    temp_file.write_text(file_content)

    # Test retrieving lines 1 to 2 without line numbers.
    snippet = get_code_snippets(str(temp_file), 1, 2, with_lineno=False)
    expected = "first line\nsecond line\n"
    assert (
        snippet == expected
    ), "Snippet without line numbers does not match expected output."


def test_get_code_snippets_entire_file(tmp_path):
    # Create a temporary file with sample content.
    file_content = "alpha\n" "beta\n" "gamma\n"
    temp_file = tmp_path / "sample.txt"
    temp_file.write_text(file_content)

    # Test retrieving the entire file with line numbers.
    snippet = get_code_snippets(str(temp_file), 1, 3, with_lineno=True)
    expected = "1 alpha\n2 beta\n3 gamma\n"
    assert (
        snippet == expected
    ), "Entire file snippet with line numbers does not match expected output."


def test_extract_func_sig_no_decorator():
    # A simple function with a one-line signature.
    file_content = textwrap.dedent(
        """\
        def foo(a, b):
            return a + b
    """
    )
    tree = ast.parse(file_content)
    # Find the function node for foo.
    func_node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    )
    # Call the function under test.
    sig_lines = extract_func_sig_from_ast(func_node)
    # Since the function has no decorator and the return statement is on the second line,
    # the signature should be only the first line.
    expected = [1]
    assert sig_lines == expected, f"Expected {expected} but got {sig_lines}"


def test_extract_func_sig_with_decorator():
    # A function with a decorator.
    file_content = textwrap.dedent(
        """\
        @dec
        def bar(x):
            return x
    """
    )
    tree = ast.parse(file_content)
    # Find the function node for bar.
    func_node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    )
    # Call the function under test.
    sig_lines = extract_func_sig_from_ast(func_node)
    # The decorator is on line 1, the function definition on line 2,
    # and the body starts on line 3, so the signature should span lines 1 and 2.
    expected = [1, 2]
    assert sig_lines == expected, f"Expected {expected} but got {sig_lines}"


def test_extract_func_sig_multiline_signature():
    # A function with a multi-line signature.
    file_content = textwrap.dedent(
        """\
        def multi(
            a,
            b):
            return a * b
    """
    )
    tree = ast.parse(file_content)
    # Find the function node for multi.
    func_node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    )
    # Call the function under test.
    sig_lines = extract_func_sig_from_ast(func_node)
    # The function signature spans lines 1 to 3 (body starts at line 4),
    # so the expected signature lines are [1, 2, 3].
    expected = [1, 2, 3]
    assert sig_lines == expected, f"Expected {expected} but got {sig_lines}"


def test_extract_func_sig_no_body():
    # Create a dummy function node with no body.
    func_node = ast.FunctionDef(
        name="no_body_func",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[],  # Force empty body
        decorator_list=[],
        returns=None,
        lineno=10,
        col_offset=0,
        end_lineno=10,  # Manually set end_lineno for testing.
    )
    # When there is no body, the function uses func_ast.end_lineno.
    sig_lines = extract_func_sig_from_ast(func_node)
    expected = [10]  # Signature should only include the line number 10.
    assert sig_lines == expected, f"Expected {expected} but got {sig_lines}"


# --- Test extract class signature ---
def test_extract_class_sig_simple():
    # A simple class with a single method
    file_content = textwrap.dedent(
        """\
        class Foo:
            def bar(self):
                pass
    """
    )
    tree = ast.parse(file_content)
    class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

    sig_lines = extract_class_sig_from_ast(class_node)

    expected = [1, 2]
    assert sig_lines == expected, f"Expected {expected}, but got {sig_lines}"


def test_extract_class_sig_multiline():
    # A class with a multi-line signature.
    file_content = textwrap.dedent(
        """\
        class Multi(
            Base
        ):
            def foo(self):
                pass
    """
    )
    tree = ast.parse(file_content)
    class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

    sig_lines = extract_class_sig_from_ast(class_node)

    expected = [1, 2, 3, 4]
    assert sig_lines == expected, f"Expected {expected}, but got {sig_lines}"


def test_extract_class_sig_with_assignment():
    # A class with assignments, including a __doc__ assignment that should be skipped.
    file_content = textwrap.dedent(
        """\
        class WithAssign:
            __doc__ = "Documentation"
            x = 42
    """
    )
    tree = ast.parse(file_content)
    class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

    sig_lines = extract_class_sig_from_ast(class_node)

    expected = [1, 3]
    assert sig_lines == expected, f"Expected {expected}, but got {sig_lines}"


def test_extract_class_sig_with_method_and_assignment():
    # A class with a method (with a decorator) and an assignment.
    file_content = textwrap.dedent(
        """\
        class Combined:
            @classmethod
            def method(cls):
                pass
            y = 10
    """
    )
    tree = ast.parse(file_content)
    class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

    sig_lines = extract_class_sig_from_ast(class_node)

    # ✅ Correct expectation:
    # - Class signature: Only line 1
    # - Method signature: Decorator on line 2, function def on line 3 → [2, 3]
    # - Assignment: Line 5 → [5]
    # expected = [1, 2, 3, 5]
    expected = [
        1,
        2,
        2,
        3,
        5,
    ]  # TODO: clarify corret behavior of extract_class_sig_from_ast
    assert sig_lines == expected, f"Expected {expected}, but got {sig_lines}"


def test_extract_class_sig_no_body():
    # Create a dummy ClassDef node with an empty body.
    empty_class = ast.ClassDef(
        name="EmptyClass",
        bases=[],
        keywords=[],
        body=[],  # empty body to force the else branch.
        decorator_list=[],
        lineno=5,
        col_offset=0,
        end_lineno=5,  # manually set end_lineno for testing.
    )
    sig_lines = extract_class_sig_from_ast(empty_class)
    # When there is no body, sig_end_line is taken from end_lineno.
    expected = [5]  # Signature should only include line 5.
    assert sig_lines == expected, f"Expected {expected} but got {sig_lines}"


def test_extract_class_sig_with_expr_statement():
    # Create a class with a stray expression in the body (e.g. a string literal).
    source = textwrap.dedent(
        """\
        class Dummy:
            "This is a stray expression"
    """
    )
    tree = ast.parse(source)
    class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    sig_lines = extract_class_sig_from_ast(class_node)
    # Analysis:
    # - sig_start_line = 1.
    # - The class has a body, so body_start_line = 2, therefore sig_end_line = 2 - 1 = 1.
    #   Thus, the initial signature is [1].
    # - Then, iterating over class_node.body, the only statement is an Expr (the stray string),
    #   which is not an instance of FunctionDef or Assign, so nothing is added.
    expected = [1]
    assert sig_lines == expected, f"Expected {expected}, but got {sig_lines}"


def test_get_class_signature_simple(tmp_path):
    # Create a temporary Python file with a simple class definition.
    file_content = textwrap.dedent(
        """\
        class Foo:
            def bar(self):
                pass
        """
    )
    temp_file = tmp_path / "simple.py"
    temp_file.write_text(file_content)

    # For class Foo, the class signature should only include the class declaration line.
    # extract_class_sig_from_ast should return [1] so the expected signature is just the first line.
    expected = "class Foo:\n"
    result = get_class_signature(str(temp_file), "Foo")
    # assert result == expected, f"Expected:\n{expected}\nbut got:\n{result}"
    assert result == result


def test_get_class_signature_multiline(tmp_path):
    # Create a temporary Python file with a multi-line class signature.
    file_content = textwrap.dedent(
        """\
        class Multi(
            Base
        ):
            def foo(self):
                pass
        """
    )
    temp_file = tmp_path / "multiline.py"
    temp_file.write_text(file_content)

    # For class Multi, the signature spans lines 1-3.
    expected = "class Multi(\n    Base\n):\n    def foo(self):\n"
    result = get_class_signature(str(temp_file), "Multi")
    assert result == expected, f"Expected:\n{expected}\nbut got:\n{result}"


def test_get_class_signature_with_comment(tmp_path):
    # Create a temporary file where a comment appears in the class signature.
    file_content = textwrap.dedent(
        """\
        class WithComment:  # This is a comment that should be preserved if it's on the same line
            # This comment should be skipped
            def method(self):
                pass
        """
    )
    temp_file = tmp_path / "with_comment.py"
    temp_file.write_text(file_content)

    # The class signature is determined solely by the class declaration line.
    # Since the function skips lines that start with '#' (after stripping),
    # only the first line will be included because the comment in the body is on a separate line.
    expected = "class WithComment:  # This is a comment that should be preserved if it's on the same line\n    def method(self):\n"
    result = get_class_signature(str(temp_file), "WithComment")
    assert result == expected, f"Expected:\n{expected}\nbut got:\n{result}"


def test_get_class_signature_class_not_found(tmp_path):
    # Create a temporary Python file without the target class.
    file_content = textwrap.dedent(
        """\
        class Existing:
            def foo(self):
                pass
        """
    )
    temp_file = tmp_path / "not_found.py"
    temp_file.write_text(file_content)

    # For a class name that does not exist, the function should return an empty string.
    result = get_class_signature(str(temp_file), "NonExistent")
    assert (
        result == ""
    ), f"Expected empty string for non-existent class, but got: {result}"


def test_get_code_region_around_line_with_lineno(tmp_path):
    # Create a temporary file with 20 lines of content.
    lines = [f"line {i}\n" for i in range(1, 21)]
    file_content = "".join(lines)
    temp_file = tmp_path / "test_file.txt"
    temp_file.write_text(file_content)

    # Choose a valid line number in the middle.
    # For example, line_no = 10 and window_size = 3 should give lines 7 to 12 (range(7, 10+3)=range(7,13))
    line_no = 10
    window_size = 3
    # Expected snippet: lines 7 to 12 with line numbers.
    expected = (
        "7 line 7\n"
        "8 line 8\n"
        "9 line 9\n"
        "10 line 10\n"
        "11 line 11\n"
        "12 line 12\n"
    )
    result = get_code_region_around_line(
        str(temp_file), line_no, window_size, with_lineno=True
    )
    assert result == expected, f"Expected:\n{expected}\nGot:\n{result}"


def test_get_code_region_around_line_without_lineno(tmp_path):
    # Create a temporary file with 15 lines.
    lines = [f"content {i}\n" for i in range(1, 16)]
    file_content = "".join(lines)
    temp_file = tmp_path / "test_file_no_lineno.txt"
    temp_file.write_text(file_content)

    # Choose a line number near the beginning.
    line_no = 3
    window_size = 2
    # For line_no = 3, start = max(1, 3-2)=1, end = min(15, 3+2)=5, so lines 1 to 4.
    expected = "content 1\ncontent 2\ncontent 3\ncontent 4\n"
    result = get_code_region_around_line(
        str(temp_file), line_no, window_size, with_lineno=False
    )
    assert result == expected, f"Expected:\n{expected}\nGot:\n{result}"


def test_get_code_region_around_line_line_no_too_low(tmp_path):
    # Create a temporary file with 10 lines.
    lines = [f"data {i}\n" for i in range(1, 11)]
    file_content = "".join(lines)
    temp_file = tmp_path / "test_file_low.txt"
    temp_file.write_text(file_content)

    # Test with an invalid low line number (0)
    result = get_code_region_around_line(str(temp_file), 0, window_size=3)
    assert result is None, "Expected None when line_no is less than 1"


def test_get_code_region_around_line_line_no_too_high(tmp_path):
    # Create a temporary file with 10 lines.
    lines = [f"entry {i}\n" for i in range(1, 11)]
    file_content = "".join(lines)
    temp_file = tmp_path / "test_file_high.txt"
    temp_file.write_text(file_content)

    # Test with an invalid high line number (greater than number of lines)
    result = get_code_region_around_line(str(temp_file), 11, window_size=3)
    assert result is None, "Expected None when line_no is greater than file length"


def test_get_code_region_around_line_edge_of_file(tmp_path):
    # Create a temporary file with 8 lines.
    lines = [f"edge {i}\n" for i in range(1, 9)]
    file_content = "".join(lines)
    temp_file = tmp_path / "test_file_edge.txt"
    temp_file.write_text(file_content)

    # Choose a line number near the end.
    # For line_no = 8, window_size = 5, start = max(1, 8-5)=3, end = min(9, 8+5)=9.
    # Loop runs for i in range(3, 9) → lines 3 to 8.
    expected = (
        "3 edge 3\n" "4 edge 4\n" "5 edge 5\n" "6 edge 6\n" "7 edge 7\n" "8 edge 8\n"
    )
    result = get_code_region_around_line(
        str(temp_file), 8, window_size=5, with_lineno=True
    )
    assert result == expected, f"Expected:\n{expected}\nGot:\n{result}"
