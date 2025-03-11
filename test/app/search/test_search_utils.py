import ast
import glob
import os
import pytest

from os.path import join as pjoin

from app.search.search_utils import is_test_file, find_python_files, parse_class_def_args

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
        "greatest_common_divisor.py", # This is not a test file, but it has "test" in its name, should not be recognized as a test file
    ]

    # Execute and verify: test files should return True, non-test files should return False
    for test_file in test_files:
        assert is_test_file(test_file), f"{test_file} should be recognized as a test file."
    for non_test_file in non_test_files:
        assert not is_test_file(non_test_file), f"{non_test_file} should not be recognized as a test file."

    
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
    assert len(python_files_rel) == len(expected_python_files), (
        f"Expected {len(expected_python_files)} python files, but got {len(python_files_rel)}."
    )
    
    # Compare each element
    for expected, actual in zip(expected_python_files, python_files_rel):
        assert actual == expected, f"Expected {expected}, but got {actual}."

def test_parse_class_def_args_simple():
    source = "class Foo(B, object):\n    pass"
    tree = ast.parse(source)
    node = tree.body[0]  # The ClassDef node for Foo
    result = parse_class_def_args(source, node)
    # 'B' is returned; 'object' is skipped.
    assert result == ["B"]

def test_parse_class_def_args_type_call():
    source = "class Bar(type('D', (), {})):\n    pass"
    tree = ast.parse(source)
    node = tree.body[0]
    result = parse_class_def_args(source, node)
    # The source segment for the first argument of the type() call is "'D'"
    assert result == ["'D'"]

def test_parse_class_def_args_mixed():
    source = "class Baz(C, type('E', (), {}), object):\n    pass"
    tree = ast.parse(source)
    node = tree.body[0]
    result = parse_class_def_args(source, node)
    # The expected bases are "C" from the ast.Name and "'E'" from the type() call.
    assert result == ["C", "'E'"]

def test_parse_class_def_args_only_object():
    source = "class Quux(object):\n    pass"
    tree = ast.parse(source)
    node = tree.body[0]
    result = parse_class_def_args(source, node)
    # Since only object is used, the result should be an empty list.
    assert result == []
    