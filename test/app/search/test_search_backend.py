import os
import pytest
import textwrap

from pathlib import Path
from tempfile import TemporaryDirectory

from app.search.search_backend import (
    LineRange,
    SearchBackend,
    SearchResult,
    RESULT_SHOW_LIMIT,
    BugLocation,
)


#######################################
### Dummy Functions ###################
#######################################
## NOTE: We use dummy functions to simulate the behavior of the functions, returning predictable results.
## These functions are used as monkey-patched implementations for the actual functions.
## In Monkey-Patching, we temporarily replace the original function with a dummy function for testing purposes.

# A fake implementation for find_python_files
def fake_find_python_files(project_path: str):
    # Return a list with a single file in the project path.
    return [os.path.join(project_path, "sample.py")]


# A fake implementation for parse_python_file
def fake_parse_python_file(py_file: str):
    # For the purpose of this test, assume the file is successfully parsed and returns:
    #  - a list of classes: (class_name, start_line, end_line)
    #  - a dict mapping class names to their methods: {class_name: [(method_name, start_line, end_line)]}
    #  - a list of top-level functions: [(func_name, start_line, end_line)]
    #  - a dict for class relation mapping: {(class_name, start_line, end_line): [list of superclasses]}
    classes = [("A", 1, 3)]
    class_to_funcs = {"A": [("method", 2, 2)]}
    top_level_funcs = [
        ("func", 4, 4)
    ]  # TODO: clarify behavior of top-level functions, should 'method' inside class A be included?
    class_relation_map = {(("A", 1, 3)): []}
    return classes, class_to_funcs, top_level_funcs, class_relation_map


# Dummy implementation to simulate _build_python_index's output.
def dummy_build_python_index(project_path: str):
    class_index = {"A": [("dummy.py", LineRange(1, 3))]}
    class_func_index = {"A": {"method": [("dummy.py", LineRange(2, 2))]}}
    function_index = {"func": [("dummy.py", LineRange(4, 4))]}
    class_relation_index = {"A": []}
    parsed_files = ["dummy.py"]
    return (
        class_index,
        class_func_index,
        function_index,
        class_relation_index,
        parsed_files,
    )

# Dummy snippet generator.
def dummy_get_code_snippets(file_name, start, end):
    return f"code from {file_name} lines {start}-{end}"

class TestSearchBackend:

    def test_build_index(self, monkeypatch):
        # Create an instance of SearchBackend with a dummy project_path.
        sb = SearchBackend(project_path="dummy_project")
        # Clear the indices to start fresh.
        sb.class_index = {}
        sb.class_func_index = {}
        sb.function_index = {}
        sb.class_relation_index = {}
        sb.parsed_files = []

        # Monkeypatch _build_python_index to use our dummy implementation.
        monkeypatch.setattr(
            SearchBackend, "_build_python_index", staticmethod(dummy_build_python_index)
        )

        # Call the _build_index method which internally updates the indices.
        sb._build_index()

        # Verify that the instance attributes have been updated as expected.
        assert sb.class_index == {"A": [("dummy.py", LineRange(1, 3))]}
        assert sb.class_func_index == {"A": {"method": [("dummy.py", LineRange(2, 2))]}}
        assert sb.function_index == {"func": [("dummy.py", LineRange(4, 4))]}
        assert sb.class_relation_index == {"A": []}
        assert sb.parsed_files == ["dummy.py"]

    def test_update_indices(self):
        # Create a SearchBackend instance with a dummy project path.
        sb = SearchBackend(project_path="dummy_project")
        # Reset indexes to empty to start with a known state.
        sb.class_index = {}
        sb.class_func_index = {}
        sb.function_index = {}
        sb.class_relation_index = (
            {}
        )  # even though originally a defaultdict, we reset for test simplicity.
        sb.parsed_files = []

        # Prepare dummy indices and parsed files.
        dummy_class_index = {"A": [("file1.py", (1, 10))]}
        dummy_class_func_index = {"A": {"method": [("file1.py", (2, 5))]}}
        dummy_function_index = {"func": [("file2.py", (20, 30))]}
        dummy_class_relation_index = {"A": ["B", "C"]}
        dummy_parsed_files = ["file1.py", "file2.py"]
        # Call _update_indices with dummy data.
        sb._update_indices(
            dummy_class_index,
            dummy_class_func_index,
            dummy_function_index,
            dummy_class_relation_index,
            dummy_parsed_files,
        )
        # Verify that the attributes have been updated as expected.
        assert sb.class_index == dummy_class_index
        assert sb.class_func_index == dummy_class_func_index
        assert sb.function_index == dummy_function_index
        assert sb.class_relation_index == dummy_class_relation_index
        assert sb.parsed_files == dummy_parsed_files

    def test_build_python_index(self, monkeypatch):
        # Create a temporary project directory with one sample Python file.
        with TemporaryDirectory() as temp_dir:
            sample_file = os.path.join(temp_dir, "sample.py")
            with open(sample_file, "w") as f:
                # Write some dummy Python content.
                f.write("class A:\n")
                f.write("    def method(self):\n")
                f.write("        pass\n")
                f.write("\n")
                f.write("def func():\n")
                f.write("    pass\n")

            # Monkey-patch the search_utils functions used in _build_python_index.
            monkeypatch.setattr(
                "app.search.search_backend.search_utils.find_python_files",
                fake_find_python_files,
            )
            monkeypatch.setattr(
                "app.search.search_backend.search_utils.parse_python_file",
                fake_parse_python_file,
            )

            # Call the _build_python_index method
            # Make sure to pass the temporary directory as the project path.
            (
                class_index,
                class_func_index,
                function_index,
                class_relation_index,
                parsed_py_files,
            ) = SearchBackend._build_python_index(temp_dir)

            # Check that the indexes were built correctly.
            # (1) Class index should contain class "A"
            assert "A" in class_index
            # And it should map to a list containing one tuple with the file and its line range.
            assert class_index["A"] == [(sample_file, LineRange(1, 3))]

            # (2) Class-function index should contain class "A" with method "method"
            assert "A" in class_func_index
            assert "method" in class_func_index["A"]
            assert class_func_index["A"]["method"] == [(sample_file, LineRange(2, 2))]

            # (3) Top-level function index should contain "func"
            assert "func" in function_index
            assert function_index["func"] == [(sample_file, LineRange(4, 4))]

            # (4) Class relation index should have "A" with an empty list for superclasses.
            assert "A" in class_relation_index
            assert class_relation_index["A"] == []

            # (5) Parsed files should list our sample file.
            assert parsed_py_files == [sample_file]

    def test_file_line_to_class_and_func(self):
        dummy_file = "dummy.py"

        # Create an instance of SearchBackend with a dummy project_path.
        sb = SearchBackend(project_path="dummy_project")

        # Set up the class-function index for a method within a class.
        # Structure: {class_name: {function_name: [(file_name, (start, end))]}}
        sb.class_func_index = {"MyClass": {"my_method": [(dummy_file, (10, 20))]}}

        # Set up the top-level function index.
        # Structure: {function_name: [(file_name, (start, end))]}
        sb.function_index = {"top_func": [(dummy_file, (30, 40))]}

        # Test case 1: Line inside a class method.
        # The line 15 should be within the method "my_method" of "MyClass".
        result = sb._file_line_to_class_and_func(dummy_file, 15)
        assert result == (
            "MyClass",
            "my_method",
        ), f"Expected ('MyClass', 'my_method'), got {result}"

        # Test case 2: Line inside a top-level function.
        # The line 35 should be within the top-level function "top_func".
        result = sb._file_line_to_class_and_func(dummy_file, 35)
        assert result == (
            None,
            "top_func",
        ), f"Expected (None, 'top_func'), got {result}"

        # Test case 3: Line not within any function.
        # The line 50 should not be captured by any index.
        result = sb._file_line_to_class_and_func(dummy_file, 50)
        assert result == (None, None), f"Expected (None, None), got {result}"

    def test_search_func_in_class(self, monkeypatch):
        # Monkey-patch the search_utils.get_code_snippets function.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            dummy_get_code_snippets,
        )

        # Create a dummy SearchBackend instance.
        sb = SearchBackend(project_path="dummy_project")

        # Set up the class_func_index with a sample entry.
        dummy_file = "dummy.py"
        sb.class_func_index = {"TestClass": {"func": [(dummy_file, (10, 20))]}}

        # Call _search_func_in_class for an existing function in a class.
        results = sb._search_func_in_class("func", "TestClass")

        # Assert one SearchResult is returned with the expected attributes.
        assert len(results) == 1
        res = results[0]
        expected_code = dummy_get_code_snippets(dummy_file, 10, 20)
        assert res.file_path == dummy_file
        assert res.start == 10
        assert res.end == 20
        assert res.class_name == "TestClass"
        assert res.func_name == "func"
        assert res.code == expected_code

        # case where class_name is not in self.class_func_index
        results = sb._search_func_in_class("func", "NonExistingClass")
        assert results == []

        # case where class_name is in, but function_name is not in self.class_func_index[class_name]
        results = sb._search_func_in_class("NonExistingFunc", "TestClass")
        assert results == []

    def test_search_func_in_all_classes(self, monkeypatch):
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            dummy_get_code_snippets,
        )

        sb = SearchBackend(project_path="dummy_project")

        dummy_file1 = "/absolute/path/file1.py"
        dummy_file2 = "/absolute/path/file2.py"

        # Set up the class_func_index with function "common" in two classes.
        sb.class_func_index = {
            "ClassA": {"common": [(dummy_file1, (5, 15))]},
            "ClassB": {"common": [(dummy_file2, (25, 35))]},
        }

        # Populate class_index so that _search_func_in_all_classes iterates over the classes.
        sb.class_index = {
            "ClassA": [(dummy_file1, (0, 100))],
            "ClassB": [(dummy_file2, (0, 100))],
        }

        # Sanity check: make sure the indexes are set up correctly.
        assert sb.class_func_index["ClassA"]["common"] == [(dummy_file1, (5, 15))]
        assert sb.class_func_index["ClassB"]["common"] == [(dummy_file2, (25, 35))]
        assert "ClassA" in sb.class_index
        assert "ClassB" in sb.class_index

        results = sb._search_func_in_all_classes("common")

        # Expect two results.
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        # Verify result from ClassA.
        res_a = next(r for r in results if r.file_path == dummy_file1)
        expected_code_a = dummy_get_code_snippets(dummy_file1, 5, 15)
        assert res_a.class_name == "ClassA"
        assert res_a.func_name == "common"
        assert res_a.code == expected_code_a
        # Verify result from ClassB.
        res_b = next(r for r in results if r.file_path == dummy_file2)
        expected_code_b = dummy_get_code_snippets(dummy_file2, 25, 35)
        assert res_b.class_name == "ClassB"
        assert res_b.func_name == "common"
        assert res_b.code == expected_code_b

    def test_search_top_level_func(self, tmp_path, monkeypatch):
        # Create a temporary file so it exists (even though our dummy won't open it).
        dummy_file = tmp_path / "top_file.py"
        dummy_file.write_text("def top_func():\n    pass\n")

        # Create a SearchBackend instance.
        sb = SearchBackend(project_path="dummy_project")

        # Set up the function_index for a top-level function.
        sb.function_index = {"top_func": [(str(dummy_file), (30, 40))]}

        # Monkey-patch the get_code_snippets function.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            dummy_get_code_snippets,
        )

        # Call the function.
        results = sb._search_top_level_func("top_func")

        # Expect one result.
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        res = results[0]

        expected_code = dummy_get_code_snippets(str(dummy_file), 30, 40)
        # For top-level functions, class_name is None.
        assert res.file_path == str(dummy_file)
        assert res.start == 30
        assert res.end == 40
        assert res.class_name is None
        assert res.func_name == "top_func"
        assert res.code == expected_code

        # case where function_name is not in self.function_index
        results = sb._search_top_level_func("NonExistingFunc")
        assert results == []

    def test_search_func_in_code_base(self, tmp_path, monkeypatch):
        from app.search.search_backend import SearchBackend, SearchResult

        # Create three temporary Python files.
        file1 = tmp_path / "file1.py"
        file1.write_text(
            textwrap.dedent(
                """\
            def top_func():
                pass
        """
            )
        )
        file2 = tmp_path / "file2.py"
        file2.write_text(
            textwrap.dedent(
                """\
            class ClassX:
                def top_func(self):
                    pass
        """
            )
        )
        file3 = tmp_path / "file3.py"
        file3.write_text(
            textwrap.dedent(
                """\
            class ClassY:
                def top_func(self):
                    pass
        """
            )
        )

        # Create a SearchBackend instance with the temporary directory as project path.
        sb = SearchBackend(project_path=str(tmp_path))

        # Set up the top-level function index.
        sb.function_index = {"top_func": [(str(file1), (1, 3))]}
        # Set up the class-function index with the same function in two classes.
        sb.class_func_index = {
            "ClassX": {"top_func": [(str(file2), (2, 4))]},
            "ClassY": {"top_func": [(str(file3), (2, 4))]},
        }
        # Populate class_index so that _search_func_in_all_classes iterates over the classes.
        sb.class_index = {
            "ClassX": [(str(file2), (1, 5))],
            "ClassY": [(str(file3), (1, 5))],
        }

        # Monkey-patch get_code_snippets with our dummy.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            dummy_get_code_snippets,
        )

        # Call the combined search function.
        results = sb._search_func_in_code_base("top_func")

        # Expect three results: one top-level and two from classes.
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # Verify top-level function result.
        res_top = next(r for r in results if r.file_path == str(file1))
        expected_top = dummy_get_code_snippets(str(file1), 1, 3)
        assert res_top.class_name is None
        assert res_top.func_name == "top_func"
        assert res_top.code == expected_top
        # Verify ClassX method result.
        res_classx = next(r for r in results if r.file_path == str(file2))
        expected_classx = dummy_get_code_snippets(str(file2), 2, 4)
        assert res_classx.class_name == "ClassX"
        assert res_classx.func_name == "top_func"
        assert res_classx.code == expected_classx
        # Verify ClassY method result.
        res_classy = next(r for r in results if r.file_path == str(file3))
        expected_classy = dummy_get_code_snippets(str(file3), 2, 4)
        assert res_classy.class_name == "ClassY"
        assert res_classy.func_name == "top_func"
        assert res_classy.code == expected_classy

    def test_get_candidate_matched_py_files(self):
        sb = SearchBackend(project_path="dummy_project")
        # Set up parsed_files with absolute paths (using various cases)
        sb.parsed_files = [
            "/abs/path/Foo.py",
            "/abs/path/bar.PY",
            "/abs/path/Baz.txt",
            "/abs/path/otherfoo.Py",
        ]
        # Test 1: Find files ending with "foo.py" (case-insensitive).
        # Expected candidates: "/abs/path/Foo.py" and "/abs/path/otherfoo.Py"
        candidates = sb._get_candidate_matched_py_files("foo.py")
        expected_candidates = {"/abs/path/Foo.py", "/abs/path/otherfoo.Py"}
        assert (
            set(candidates) == expected_candidates
        ), f"Expected {expected_candidates}, got {candidates}"

        # Test 2: Find files ending with "BAR.py" (case-insensitive).
        candidates = sb._get_candidate_matched_py_files("BAR.py")
        expected_candidates = {"/abs/path/bar.PY"}
        assert (
            set(candidates) == expected_candidates
        ), f"Expected {expected_candidates}, got {candidates}"

        # Test 3: No matching files should return an empty list.
        candidates = sb._get_candidate_matched_py_files("nonexistent.py")
        assert candidates == [], f"Expected empty list, got {candidates}"

    #######################################
    ### Testing Interfaces ################
    #######################################

    def test_get_class_full_snippet_not_found(self):
        # Create a SearchBackend instance with an empty class_index.
        sb = SearchBackend(project_path="dummy_project")
        sb.class_index = {}  # ensure no classes are indexed

        # Call get_class_full_snippet with a class name that doesn't exist.
        result, search_res, flag = sb.get_class_full_snippet("NonExisting")
        # Expect a message indicating that the class was not found, no search results, and flag False.
        expected_message = "Could not find class NonExisting in the codebase."
        assert result == expected_message
        assert search_res == []
        assert flag is False

    def test_get_class_full_snippet_empty_class_index(self):
        # Create a SearchBackend instance with an empty list for class "A".
        sb = SearchBackend(project_path="dummy_project")
        sb.class_index = {"A": []}  # key exists, but no occurrences

        # Call get_class_full_snippet for class "A".
        result, search_res, flag = sb.get_class_full_snippet("A")

        # Expect a message indicating that the class was not found, no search results, and flag False.
        expected_message = "Could not find class A in the codebase."
        assert result == expected_message
        assert search_res == []
        assert flag is False

    def test_get_class_full_snippet_too_many_results(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")

        # Set up class_index with three occurrences of class "A"
        sb.class_index = {
            "A": [
                ("/absolute/path/fileA.py", (1, 10)),
                ("/absolute/path/fileB.py", (11, 20)),
                ("/absolute/path/fileC.py", (21, 30)),
            ]
        }

        # Monkey-patch get_code_snippets to return a dummy snippet.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            lambda file_path, start, end, with_lineno=True: f"dummy code snippet from {file_path} lines {start}-{end}",
        )

        # Monkey-patch SearchResult.to_tagged_str to return a predictable string.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, project_path: f"tagged snippet from {self.file_path} {self.start}-{self.end}",
        )

        # Call get_class_full_snippet for class "A".
        result, search_res, flag = sb.get_class_full_snippet("A")

        # Verify that flag is True
        assert flag is True

        # Verify that the search results are truncated to 2 even though there are 3 entries.
        assert len(search_res) == 2

        # Verify that the message for too many results is included.
        assert "Too many results, showing full code for 2 of them:" in result

    def test_get_class_full_snippet_found(self, monkeypatch):

        sb = SearchBackend(project_path="dummy_project")

        # Set up class_index with a sample class "A" and one occurrence.
        sb.class_index = {"A": [("/absolute/path/fileA.py", (1, 10))]}

        # Monkey-patch get_code_snippets so it returns a predictable dummy snippet.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            lambda file_path, start, end, with_lineno=True: f"dummy code snippet from {file_path} lines {start}-{end}",
        )

        # Monkey-patch SearchResult.to_tagged_str to return a predictable tagged string.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, project_path: f"tagged snippet from {self.file_path} {self.start}-{self.end}",
        )

        # Call get_class_full_snippet for class "A".
        result, search_res, flag = sb.get_class_full_snippet("A")

        # Verify that flag is True and there is one SearchResult.
        assert flag is True
        assert len(search_res) == 1

        # Verify that the result message starts with the expected header.
        expected_header = "Found 1 classes with name A in the codebase:"
        assert result.startswith(expected_header)

        # Check that the tagged snippet from our dummy SearchResult appears in the output.
        expected_tagged = "tagged snippet from /absolute/path/fileA.py 1-10"
        assert expected_tagged in result

    def test_search_class_not_found(self):
        from app.search.search_backend import SearchBackend

        # Create a SearchBackend instance with an empty class_index.
        sb = SearchBackend(project_path="dummy_project")
        sb.class_index = {}  # No classes indexed.

        # Call search_class with a class name that doesn't exist.
        result, search_res, flag = sb.search_class("NonExisting")

        # Verify that the error message, empty result list, and False flag are returned.
        expected_message = "Could not find class NonExisting in the codebase."
        assert result == expected_message
        assert search_res == []
        assert flag is False

    def test_search_class_empty_search_results(self):
        # Create a SearchBackend instance with a key for "A" but with an empty list.
        sb = SearchBackend(project_path="dummy_project")
        sb.class_index = {"A": []}  # key exists, but no occurrences

        # Call search_class for "A".
        result, search_res, flag = sb.search_class("A")

        # We expect the function to return the not-found message.
        expected_message = "Could not find class A in the codebase."
        assert result == expected_message
        assert search_res == []
        assert flag is False

    def test_search_class_too_many_results(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")

        # Set up the class_index with three occurrences for class "A".
        sb.class_index = {
            "A": [
                ("/absolute/path/fileA.py", (1, 10)),
                ("/absolute/path/fileB.py", (11, 20)),
                ("/absolute/path/fileC.py", (21, 30)),
            ]
        }

        # Monkey-patch get_class_signature to return a dummy signature.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_class_signature",
            lambda fname, class_name: f"signature of {class_name} from {fname}",
        )

        # Monkey-patch collapse_to_file_level to return a predictable collapsed string.
        monkeypatch.setattr(
            SearchResult,
            "collapse_to_file_level",
            lambda search_res, project_path: "\n".join(
                f"{res.file_path}: lines {res.start}-{res.end}" for res in search_res
            ),
        )

        # Monkey-patch RESULT_SHOW_LIMIT to force the branch.
        monkeypatch.setattr("app.search.search_backend.RESULT_SHOW_LIMIT", 1)

        # Call search_class for class "A".
        result, final_search_res, flag = sb.search_class("A")

        # Check that flag is True.
        assert flag is True

        # Verify that the branch for too many results was taken by checking the message.
        assert "They appeared in the following files:" in result

        # Only RESULT_SHOW_LIMIT (1) result should be returned.
        assert len(final_search_res) == 1

    def test_search_class_found_single(self, monkeypatch):
        from app.search.search_backend import (
            SearchBackend,
            SearchResult,
            RESULT_SHOW_LIMIT,
        )

        sb = SearchBackend(project_path="dummy_project")

        # Set up class_index with one occurrence of class "MyClass".
        sb.class_index = {"MyClass": [("/absolute/path/fileA.py", (1, 10))]}

        # Override get_class_signature to return a predictable signature.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_class_signature",
            lambda fname, cname: f"Signature for {cname} in {fname}",
        )

        # Override SearchResult.to_tagged_str to return a predictable tagged string.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, project_path: f"Tagged signature for {self.file_path}",
        )

        # Call search_class for "MyClass".
        result, search_res, flag = sb.search_class("MyClass")

        # Verify the flag is True and we have one search result.
        assert flag is True
        assert len(search_res) == 1

        # The output message should start with a header indicating 1 found.
        expected_header = "Found 1 classes with name MyClass in the codebase:"
        assert result.startswith(expected_header)

        # Verify that the tagged string appears in the output.
        expected_tagged = "Tagged signature for /absolute/path/fileA.py"
        assert expected_tagged in result

    def test_search_class_in_file_file_not_found(self):
        from app.search.search_backend import SearchBackend

        sb = SearchBackend(project_path="dummy_project")

        # Simulate that no candidate file is found.
        sb._get_candidate_matched_py_files = lambda fname: []

        # Call search_class_in_file with a file name that isn't found.
        tool_output, search_res, flag = sb.search_class_in_file(
            "MyClass", "nonexistent.py"
        )

        expected_message = "Could not find file nonexistent.py in the codebase."
        assert (
            tool_output == expected_message
        ), f"Expected: {expected_message}, got: {tool_output}"
        assert search_res == []
        assert flag is False

    def test_search_class_in_file_class_not_found(self):
        from app.search.search_backend import SearchBackend

        sb = SearchBackend(project_path="dummy_project")

        # Simulate that a candidate file exists.
        candidate_file = "/abs/path/existing.py"
        sb._get_candidate_matched_py_files = lambda fname: [candidate_file]

        # Simulate that the class_index does not contain the target class.
        sb.class_index = {}

        tool_output, search_res, flag = sb.search_class_in_file(
            "MyClass", "existing.py"
        )

        expected_message = "Could not find class MyClass in the codebase."
        assert (
            tool_output == expected_message
        ), f"Expected: {expected_message}, got: {tool_output}"
        assert search_res == []
        assert flag is False

    def test_search_class_in_file_no_occurrence_in_candidate(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        # Simulate candidate file found.
        monkeypatch.setattr(
            sb,
            "_get_candidate_matched_py_files",
            lambda file_name: ["/absolute/path/dummy.py"],
        )

        # Set class_index with an occurrence in a different file.
        sb.class_index = {"MyClass": [("/absolute/path/other.py", (50, 60))]}

        # Patch get_code_snippets (though it won't be called in this branch).
        monkeypatch.setattr(
            "app.search.search_utils.get_code_snippets",
            lambda fname, start, end, with_lineno=True: "dummy class snippet",
        )

        result, search_res, flag = sb.search_class_in_file("MyClass", "dummy.py")

        # Since no occurrence matches the candidate file, we expect an error.
        expected_message = "Could not find class MyClass in file dummy.py."
        assert result == expected_message
        assert search_res == []
        assert flag is False

    def test_search_class_in_file_found(self, monkeypatch):
        from app.search.search_backend import SearchBackend, SearchResult

        sb = SearchBackend(project_path="dummy_project")

        candidate_file = "/abs/path/existing.py"
        # Simulate that candidate file is found.
        sb._get_candidate_matched_py_files = lambda fname: [candidate_file]

        # Set up class_index with the target class in the candidate file.
        sb.class_index = {"MyClass": [(candidate_file, (1, 20))]}

        # Monkey-patch get_code_snippets to return a predictable dummy code.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            lambda file_path, start, end, with_lineno=True: f"dummy class code from {file_path} lines {start}-{end}",
        )

        # Monkey-patch SearchResult.to_tagged_str to return a predictable string.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, project_path: f"Tagged class snippet from {self.file_path} {self.start}-{self.end}",
        )

        tool_output, search_res, flag = sb.search_class_in_file(
            "MyClass", "existing.py"
        )

        # Expect a header indicating 1 class was found.
        expected_header = "Found 1 classes with name MyClass in file existing.py:"
        assert tool_output.startswith(
            expected_header
        ), f"Output did not start with expected header. Got: {tool_output}"

        # Verify that one SearchResult is returned and flag is True.
        assert len(search_res) == 1, f"Expected 1 result, got: {len(search_res)}"
        assert flag is True

        # Check that the dummy code appears in the SearchResult.
        expected_dummy = "dummy class code from /abs/path/existing.py lines 1-20"
        assert (
            expected_dummy in search_res[0].code
        ), f"Expected dummy code to be in result. Got: {search_res[0].code}"

    def test_search_method_in_file_file_not_found(self):
        from app.search.search_backend import SearchBackend

        sb = SearchBackend(project_path="dummy_project")

        # Simulate no matching file found.
        sb.parsed_files = (
            []
        )  # Ensures _get_candidate_matched_py_files returns an empty list.

        # Call search_method_in_file with any method and file name.
        tool_output, results, flag = sb.search_method_in_file(
            "some_method", "nonexistent.py"
        )

        expected_message = "Could not find file nonexistent.py in the codebase."
        assert (
            tool_output == expected_message
        ), f"Expected message: {expected_message}, got: {tool_output}"
        assert results == []
        assert flag is False

    def test_search_method_in_file_filtered_res_empty(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")

        # Simulate candidate file found.
        candidate_file = "/absolute/path/dummy.py"
        monkeypatch.setattr(
            sb, "_get_candidate_matched_py_files", lambda file_name: [candidate_file]
        )

        # Patch the search function to return a result in a different file.
        def fake_search_func(method_name):
            # Return a SearchResult with a file_path that is not in the candidate list.
            return [
                SearchResult(
                    "/absolute/path/other.py",
                    100,
                    110,
                    None,
                    method_name,
                    "dummy method snippet",
                )
            ]

        monkeypatch.setattr(sb, "_search_func_in_code_base", fake_search_func)

        method_name = "nonexistent_method_in_file"
        file_name = "dummy.py"
        result, filtered_res, flag = sb.search_method_in_file(method_name, file_name)

        expected_message = (
            f"There is no method with name `{method_name}` in file {file_name}."
        )
        assert result == expected_message
        assert filtered_res == []
        assert flag is False

    def test_search_method_in_file_method_not_found(self, monkeypatch):
        from app.search.search_backend import SearchBackend

        sb = SearchBackend(project_path="dummy_project")

        # Simulate that the candidate file is found.
        candidate_file = "/abs/path/existing.py"
        sb.parsed_files = [candidate_file]

        # Override _get_candidate_matched_py_files to return the candidate file.
        monkeypatch.setattr(
            sb, "_get_candidate_matched_py_files", lambda filename: [candidate_file]
        )

        # Override _search_func_in_code_base to return an empty list (i.e. method not found anywhere).
        monkeypatch.setattr(sb, "_search_func_in_code_base", lambda method: [])

        tool_output, results, flag = sb.search_method_in_file(
            "missing_method", "existing.py"
        )

        expected_message = "The method missing_method does not appear in the codebase."
        assert (
            tool_output == expected_message
        ), f"Expected message: {expected_message}, got: {tool_output}"
        assert results == []
        assert flag is False

    def test_search_method_in_file_found(self, monkeypatch):
        from app.search.search_backend import SearchBackend, SearchResult

        sb = SearchBackend(project_path="dummy_project")

        candidate_file = "/abs/path/existing.py"
        sb.parsed_files = [candidate_file]

        # Override _get_candidate_matched_py_files to return the candidate file.
        monkeypatch.setattr(
            sb, "_get_candidate_matched_py_files", lambda filename: [candidate_file]
        )

        # Create a dummy SearchResult instance.
        # We'll simulate two results (even though the code doesn't trim results in this case).
        dummy_result1 = SearchResult(
            candidate_file, 10, 20, None, "test_method", "dummy code 1"
        )
        dummy_result2 = SearchResult(
            candidate_file, 30, 40, None, "test_method", "dummy code 2"
        )

        # Override _search_func_in_code_base to return two results.
        monkeypatch.setattr(
            sb,
            "_search_func_in_code_base",
            lambda method: [dummy_result1, dummy_result2],
        )

        # Monkey-patch SearchResult.to_tagged_str to return a predictable string.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, proj_path: f"Tagged result for {self.file_path} from {self.start} to {self.end}",
        )

        tool_output, results, flag = sb.search_method_in_file(
            "test_method", "existing.py"
        )

        # Expect that we found 2 results and flag is True.
        assert flag is True
        assert len(results) == 2

        expected_header = (
            "Found 2 methods with name `test_method` in file existing.py:\n\n"
        )
        assert tool_output.startswith(expected_header)

        # Check that each SearchResult's tagged string appears in the tool_output.
        expected_tagged1 = f"Tagged result for {candidate_file} from 10 to 20"
        expected_tagged2 = f"Tagged result for {candidate_file} from 30 to 40"
        assert (
            expected_tagged1 in tool_output
        ), f"Expected '{expected_tagged1}' in output."
        assert (
            expected_tagged2 in tool_output
        ), f"Expected '{expected_tagged2}' in output."

    def test_search_method_in_class_class_not_found(self):
        sb = SearchBackend(project_path="dummy_project")
        sb.class_index = {}  # No classes indexed.
        tool_output, results, flag = sb.search_method_in_class(
            "any_method", "MissingClass"
        )
        expected = "Could not find class MissingClass in the codebase."
        assert tool_output == expected
        assert results == []
        assert flag is False

    def test_search_method_in_class_method_not_found(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        # Set up class_index so the class exists.
        sb.class_index = {"MyClass": [("dummy.py", (1, 10))]}
        # Monkey-patch _search_func_in_class to return no results.
        monkeypatch.setattr(sb, "_search_func_in_class", lambda method, cls: [])
        tool_output, results, flag = sb.search_method_in_class("nonexistent", "MyClass")
        expected = "Could not find method nonexistent in class MyClass`."
        assert tool_output == expected
        assert results == []
        assert flag is False

    def test_search_method_in_class_single_result(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        # Set up a valid class.
        sb.class_index = {"MyClass": [("dummy.py", (1, 10))]}
        # Create a dummy SearchResult.
        dummy_result = SearchResult(
            "dummy.py", 5, 5, "MyClass", "test_method", "dummy code"
        )
        # Monkey-patch _search_func_in_class to return one search result.
        monkeypatch.setattr(
            sb, "_search_func_in_class", lambda method, cls: [dummy_result]
        )
        # Patch to_tagged_str for predictable output.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, proj: f"tagged: {self.file_path} {self.start}-{self.end}",
        )

        tool_output, results, flag = sb.search_method_in_class("test_method", "MyClass")
        expected_header = "Found 1 methods with name test_method in class MyClass:\n\n"
        assert tool_output.startswith(expected_header)
        assert "tagged: dummy.py 5-5" in tool_output
        assert results == [dummy_result]
        assert flag is True

    def test_search_method_in_class_multiple_results(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        sb.class_index = {"MyClass": [("dummy.py", (1, 10))]}
        # Create dummy results more than RESULT_SHOW_LIMIT.
        dummy_results = [
            SearchResult("dummy.py", i, i, "MyClass", "test_method", f"dummy code {i}")
            for i in range(1, RESULT_SHOW_LIMIT + 3)
        ]
        monkeypatch.setattr(
            sb, "_search_func_in_class", lambda method, cls: dummy_results
        )
        monkeyatch_tagged = (
            lambda self, proj: f"tagged: {self.file_path} {self.start}-{self.end}"
        )
        monkeypatch.setattr(SearchResult, "to_tagged_str", monkeyatch_tagged)
        # Patch collapse_to_file_level to return a predictable string.
        monkeypatch.setattr(
            SearchResult,
            "collapse_to_file_level",
            lambda results, proj: "collapsed files",
        )

        tool_output, results, flag = sb.search_method_in_class("test_method", "MyClass")
        expected_header = f"Found {len(dummy_results)} methods with name test_method in class MyClass:\n\n"
        assert tool_output.startswith(expected_header)
        # Check if extra-results message is included.
        assert "Too many results, showing full code for" in tool_output
        assert "Other results are in these files:" in tool_output
        assert "collapsed files" in tool_output
        # Verify that only the first RESULT_SHOW_LIMIT results are returned.
        assert results == dummy_results[:RESULT_SHOW_LIMIT]
        assert flag is True

    def test_search_method_not_found(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        # Monkey-patch _search_func_in_code_base to simulate no results.
        monkeypatch.setattr(sb, "_search_func_in_code_base", lambda method: [])
        tool_output, results, flag = sb.search_method("missing_method")
        expected = "Could not find method missing_method in the codebase."
        assert tool_output == expected
        assert results == []
        assert flag is False

    def test_search_method_with_results_within_limit(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        # Create a dummy result list within the RESULT_SHOW_LIMIT.
        dummy_result = SearchResult(
            "dummy.py", 10, 20, None, "test_method", "dummy code"
        )
        dummy_results = [dummy_result]
        monkeypatch.setattr(
            sb, "_search_func_in_code_base", lambda method: dummy_results
        )
        # Patch to_tagged_str for predictable tagged output.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, proj: f"tagged: {self.file_path} {self.start}-{self.end}",
        )

        tool_output, results, flag = sb.search_method("test_method")
        expected_header = "Found 1 methods with name test_method in the codebase:\n\n"
        assert tool_output.startswith(expected_header)
        assert "tagged: dummy.py 10-20" in tool_output
        assert results == dummy_results
        assert flag is True

    def test_search_method_exceeding_limit(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        # Create dummy results exceeding RESULT_SHOW_LIMIT.
        dummy_results = [
            SearchResult("dummy.py", i, i, None, "test_method", f"dummy code {i}")
            for i in range(1, RESULT_SHOW_LIMIT + 2)
        ]
        monkeypatch.setattr(
            sb, "_search_func_in_code_base", lambda method: dummy_results
        )
        # Patch collapse_to_file_level to return predictable collapsed file names.
        monkeypatch.setattr(
            SearchResult,
            "collapse_to_file_level",
            lambda results, proj: "collapsed files",
        )

        tool_output, results, flag = sb.search_method("test_method")
        expected_header = f"Found {len(dummy_results)} methods with name test_method in the codebase:\n\n"
        assert tool_output.startswith(expected_header)
        # Verify that the output indicates results exceed the show limit.
        assert "They appeared in the following files:" in tool_output
        assert "collapsed files" in tool_output
        # Only the first RESULT_SHOW_LIMIT results are returned.
        assert results == dummy_results[:RESULT_SHOW_LIMIT]
        assert flag is True

    def test_search_code_not_found(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        # Ensure at least one file is in parsed_files.
        sb.parsed_files = ["dummy.py"]
        # Patch get_code_region_containing_code to return no snippets.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_containing_code",
            lambda file_path, code_str: [],
        )
        tool_output, results, flag = sb.search_code("search_target")
        expected = "Could not find code search_target in the codebase."
        assert tool_output == expected
        assert results == []
        assert flag is False

    def test_search_code_within_limit(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        sb.parsed_files = ["dummy.py"]
        # Patch get_code_region_containing_code to return a single snippet.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_containing_code",
            lambda file_path, code_str: [(20, "dummy snippet at line 20")],
        )
        # Patch _file_line_to_class_and_func to return predictable class/method.
        monkeypatch.setattr(
            sb,
            "_file_line_to_class_and_func",
            lambda file, line: ("MyClass", "my_method"),
        )
        # Patch to_tagged_str for predictable output.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, proj: f"tagged: {self.file_path} {self.start}-{self.end}",
        )
        tool_output, results, flag = sb.search_code("search_target")
        expected_header = (
            "Found 1 snippets containing `search_target` in the codebase:\n\n"
        )
        assert tool_output.startswith(expected_header)
        assert "- Search result 1:" in tool_output
        assert "tagged: dummy.py 20-20" in tool_output
        assert results[0].file_path == "dummy.py"
        assert results[0].start == 20
        assert results[0].end == 20
        assert results[0].code == "dummy snippet at line 20"
        assert results[0].class_name == "MyClass"
        assert results[0].func_name == "my_method"
        assert flag is True

    def test_search_code_exceeding_limit(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        sb.parsed_files = ["dummy.py"]
        total_results = RESULT_SHOW_LIMIT + 2
        # Create multiple dummy snippets.
        dummy_snippets = [
            (i + 10, f"dummy snippet {i + 10}") for i in range(total_results)
        ]
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_containing_code",
            lambda file_path, code_str: dummy_snippets,
        )
        # Patch _file_line_to_class_and_func to return a constant dummy value.
        monkeypatch.setattr(
            sb,
            "_file_line_to_class_and_func",
            lambda file, line: ("MyClass", "my_method"),
        )
        # Patch collapse_to_file_level to return a predictable string.
        monkeypatch.setattr(
            SearchResult,
            "collapse_to_file_level",
            lambda results, proj: "collapsed files",
        )
        tool_output, results, flag = sb.search_code("search_target")
        expected_header = f"Found {total_results} snippets containing `search_target` in the codebase:\n\n"
        assert tool_output.startswith(expected_header)
        # Should use file-level collapse since total results exceed the limit.
        assert "They appeared in the following files:" in tool_output
        assert "collapsed files" in tool_output
        # Only first RESULT_SHOW_LIMIT results are returned.
        assert results == results[:RESULT_SHOW_LIMIT]
        assert len(results) == RESULT_SHOW_LIMIT
        assert flag is True

    def test_search_code_in_file_file_not_found(self):
        sb = SearchBackend(project_path="dummy_project")
        sb.parsed_files = ["dummy.py"]  # Only file available does not match target.
        tool_output, results, flag = sb.search_code_in_file(
            "sample_code)", "nonexistent.py"
        )
        expected = "Could not find file nonexistent.py in the codebase."
        assert tool_output == expected
        assert results == []
        assert flag is False

    def test_search_code_in_file_code_not_found(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        sb.parsed_files = ["existing.py"]
        # Patch get_code_region_containing_code to return no snippets.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_containing_code",
            lambda file_path, code: [],
        )
        # The input code_str "snippet)" becomes "snippet" after removalsuffix.
        tool_output, results, flag = sb.search_code_in_file("snippet)", "existing.py")
        expected = "Could not find code snippet in file existing.py."
        assert tool_output == expected
        assert results == []
        assert flag is False

    def test_search_code_in_file_within_limit(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        sb.parsed_files = ["/abs/path/existing.py"]
        # Patch get_code_region_containing_code to return one dummy snippet.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_containing_code",
            lambda file_path, code: [(50, "dummy code at line 50")],
        )
        # Patch _file_line_to_class_and_func to return predictable values.
        monkeypatch.setattr(
            sb,
            "_file_line_to_class_and_func",
            lambda file, line: ("MyClass", "my_method"),
        )
        # Patch to_tagged_str for predictable output.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, proj: f"tagged: {self.file_path} {self.start}-{self.end}",
        )
        tool_output, results, flag = sb.search_code_in_file("snippet)", "existing.py")
        expected_header = "Found 1 snippets with code snippet in file existing.py:\n\n"
        assert tool_output.startswith(expected_header)
        assert "tagged: /abs/path/existing.py 50-50" in tool_output
        res = results[0]
        assert res.file_path == "/abs/path/existing.py"
        assert res.start == 50
        assert res.end == 50
        assert res.code == "dummy code at line 50"
        assert res.class_name == "MyClass"
        assert res.func_name == "my_method"
        assert flag is True

    def test_search_code_in_file_exceeding_limit(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        sb.parsed_files = ["/abs/path/existing.py"]
        total_results = RESULT_SHOW_LIMIT + 3
        # Create multiple dummy snippets.
        dummy_snippets = [
            (i + 100, f"dummy code at line {i + 100}") for i in range(total_results)
        ]
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_containing_code",
            lambda file_path, code: dummy_snippets,
        )
        # Patch _file_line_to_class_and_func to return constant dummy values.
        monkeypatch.setattr(
            sb,
            "_file_line_to_class_and_func",
            lambda file, line: ("MyClass", "my_method"),
        )
        # Patch collapse_to_method_level to return a predictable string.
        monkeypatch.setattr(
            SearchResult,
            "collapse_to_method_level",
            lambda results, proj: "collapsed methods",
        )
        tool_output, results, flag = sb.search_code_in_file("snippet)", "existing.py")
        expected_header = (
            f"Found {total_results} snippets with code snippet in file existing.py:\n\n"
        )
        assert tool_output.startswith(expected_header)
        # Check that file-level collapse is used.
        assert "They appeared in the following methods:" in tool_output
        assert "collapsed methods" in tool_output
        # Only the first RESULT_SHOW_LIMIT results are returned.
        assert len(results) == RESULT_SHOW_LIMIT
        assert results == results[:RESULT_SHOW_LIMIT]
        assert flag is True

    def test_get_code_around_line_file_not_found(self):
        sb = SearchBackend(project_path="dummy_project")
        # Simulate no candidate files found.
        sb._get_candidate_matched_py_files = lambda fname: []

        tool_output, results, flag = sb.get_code_around_line(
            "nonexistent.py", "10", "2"
        )
        expected = "Could not find file nonexistent.py in the codebase."
        assert tool_output == expected
        assert results == []
        assert flag is False

    def test_get_code_around_line_invalid_line(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")

        # Simulate candidate file(s) being found.
        monkeypatch.setattr(
            sb,
            "_get_candidate_matched_py_files",
            lambda file_name: ["/absolute/path/dummy.py"],
        )

        # Force get_code_region_around_line to always return None.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_around_line",
            lambda file_path, line_no, window_size: None,
        )

        # Call get_code_around_line. Since snippet is None, no region search result is added.
        result, func_search_results, flag = sb.get_code_around_line(
            "dummy.py", "10", "2"
        )

        # Expect the error message indicating the line number is invalid.
        expected_message = "10 is invalid in file dummy.py."
        assert result == expected_message
        assert func_search_results == []
        assert flag is False

    def test_get_code_around_line_with_class_and_func(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")

        # Return one candidate file.
        monkeypatch.setattr(
            sb,
            "_get_candidate_matched_py_files",
            lambda file_name: ["/absolute/path/dummy.py"],
        )

        # Simulate a valid code snippet.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_around_line",
            lambda file_path, line_no, window_size: f"dummy snippet for {file_path}",
        )

        # Simulate that the file and line correspond to a class and function.
        monkeypatch.setattr(
            sb,
            "_file_line_to_class_and_func",
            lambda file_path, line_no: ("DummyClass", "dummy_func"),
        )

        # Simulate search_method_in_class returning a dummy result.
        monkeypatch.setattr(
            sb,
            "search_method_in_class",
            lambda func_name, class_name: (
                "",
                [
                    SearchResult(
                        "/absolute/path/dummy.py",
                        1,
                        20,
                        "DummyClass",
                        "dummy_func",
                        "signature",
                    )
                ],
                True,
            ),
        )

        # Patch to_tagged_str for predictable output.
        monkeyatch_str = (
            lambda self, project_path: f"tagged snippet from {self.file_path} lines {self.start}-{self.end}"
        )
        monkeypatch.setattr(SearchResult, "to_tagged_str", monkeyatch_str)

        result, func_search_results, flag = sb.get_code_around_line(
            "dummy.py", "10", "2"
        )

        # Verify that a region search result is added.
        assert "Found 1 code snippets around line 10:" in result
        # The function search result should come from search_method_in_class.
        assert len(func_search_results) == 1
        assert flag is True

    def test_get_code_around_line_with_only_func(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")

        monkeypatch.setattr(
            sb,
            "_get_candidate_matched_py_files",
            lambda file_name: ["/absolute/path/dummy.py"],
        )

        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_around_line",
            lambda file_path, line_no, window_size: f"dummy snippet for {file_path}",
        )

        # Simulate that only a function is found (class is None).
        monkeypatch.setattr(
            sb,
            "_file_line_to_class_and_func",
            lambda file_path, line_no: (None, "dummy_func"),
        )

        # Simulate search_method returning a dummy search result.
        monkeypatch.setattr(
            sb,
            "search_method",
            lambda func_name: (
                "",
                [
                    SearchResult(
                        "/absolute/path/dummy.py",
                        1,
                        20,
                        None,
                        "dummy_func",
                        "signature",
                    )
                ],
                True,
            ),
        )

        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, project_path: f"tagged snippet from {self.file_path} lines {self.start}-{self.end}",
        )

        result, func_search_results, flag = sb.get_code_around_line(
            "dummy.py", "10", "2"
        )

        assert "Found 1 code snippets around line 10:" in result
        # The function search result should come from search_method.
        assert len(func_search_results) == 1
        assert flag is True

    def test_get_code_around_line_with_no_func_and_class(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")

        monkeypatch.setattr(
            sb,
            "_get_candidate_matched_py_files",
            lambda file_name: ["/absolute/path/dummy.py"],
        )

        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_around_line",
            lambda file_path, line_no, window_size: f"dummy snippet for {file_path}",
        )

        # Simulate that neither class nor function is found.
        monkeypatch.setattr(
            sb, "_file_line_to_class_and_func", lambda file_path, line_no: (None, None)
        )

        # No need to patch search_method or search_method_in_class because they won't be called.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, project_path: f"tagged snippet from {self.file_path} lines {self.start}-{self.end}",
        )

        result, func_search_results, flag = sb.get_code_around_line(
            "dummy.py", "10", "2"
        )

        assert "Found 1 code snippets around line 10:" in result
        # In this case, since there's no function found, func_search_results should remain empty.
        assert func_search_results == []
        assert flag is True

    def test_get_code_around_line_good(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        dummy_file = "/dummy/path/existing.py"
        # Simulate candidate file found.
        sb._get_candidate_matched_py_files = lambda fname: [dummy_file]

        # Patch get_code_region_around_line to return a dummy snippet.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_region_around_line",
            lambda file_path, line_no, window: f"Snippet of {file_path} at line {line_no} ± {window}",
        )
        # Patch _file_line_to_class_and_func to return fixed class and method.
        monkeypatch.setattr(
            sb,
            "_file_line_to_class_and_func",
            lambda file, line: ("MyClass", "my_method"),
        )
        # Create a dummy SearchResult to be returned by search_method_in_class.
        dummy_sr = SearchResult(
            dummy_file, 8, 12, "MyClass", "my_method", "dummy function code"
        )
        monkeypatch.setattr(
            sb,
            "search_method_in_class",
            lambda func, cls: ("Found function", [dummy_sr], True),
        )
        # Patch to_tagged_str for predictable output.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, proj: f"Tagged: {self.file_path} {self.start}-{self.end}",
        )

        line_no = 10
        window_size = 2
        tool_output, func_results, flag = sb.get_code_around_line(
            "existing.py", str(line_no), str(window_size)
        )

        expected_header = f"Found 1 code snippets around line {line_no}:\n\n"
        assert tool_output.startswith(expected_header)
        # Check that the tagged region appears in the output.
        assert "Tagged: /dummy/path/existing.py 8-12" in tool_output
        # Verify that the function search results include our dummy SearchResult.
        assert func_results == [dummy_sr]
        assert flag is True

    def test_get_file_content_file_not_found(self):
        sb = SearchBackend(project_path="dummy_project")
        # Set parsed_files so that no file ends with the given target.
        sb.parsed_files = ["other.py"]
        tool_output, results, flag = sb.get_file_content("nonexistent.py")
        expected = "Could not find file nonexistent.py in the codebase."
        assert tool_output == expected
        assert results == []
        assert flag is False

    def test_get_file_content_found(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        candidate_file = "/dummy/path/existing.py"
        # Simulate that parsed_files contains a matching file.
        sb.parsed_files = [candidate_file]

        # Patch Path.read_text to return dummy file content.
        from pathlib import Path

        monkeypatch.setattr(Path, "read_text", lambda self: "line1\nline2\nline3")

        tool_output, results, flag = sb.get_file_content("existing.py")

        expected_output = f"<file>existing.py</file> <code>line1\nline2\nline3</code>"
        assert tool_output == expected_output
        # Validate that a single SearchResult is returned with correct content.
        assert len(results) == 1
        sr = results[0]
        assert sr.file_path == candidate_file
        # Verify start and end line numbers.
        assert sr.start == 1
        assert sr.end == 3
        # Check that the file content is correct.
        assert sr.code == "line1\nline2\nline3"
        assert flag is True

    def test_get_file_content_file_not_found(self):
        sb = SearchBackend(project_path="dummy_project")
        sb.parsed_files = ["other.py"]  # No file ends with "missing.py"
        tool_output, results, flag = sb.get_file_content("missing.py")
        expected = "Could not find file missing.py in the codebase."
        assert tool_output == expected
        assert results == []
        assert flag is False

    def test_get_file_content_found(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        dummy_path = "/dummy/path/existing.py"
        sb.parsed_files = [dummy_path]
        # Patch Path.read_text to return dummy content.
        from pathlib import Path
        monkeypatch.setattr(Path, "read_text", lambda self: "line1\nline2\nline3")
        tool_output, results, flag = sb.get_file_content("existing.py")
        expected = f"<file>existing.py</file> <code>line1\nline2\nline3</code>"
        assert tool_output == expected
        assert len(results) == 1
        sr = results[0]
        assert sr.file_path == dummy_path
        assert sr.start == 1
        assert sr.end == 3
        assert sr.code == "line1\nline2\nline3"
        assert flag is True

    def test_retrieve_class_context_no_definition(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        # Simulate search_class_in_file always returning call_ok=False.
        monkeypatch.setattr(
            sb, "search_class_in_file", lambda **kwargs: ("", [], False)
        )
        result = sb.retrieve_class_context({("MyClass", "file.py")})
        assert result is None

    def test_retrieve_class_context_found(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        # Simulate search_class_in_file returning a definition.
        dummy_code = "class MyClass: pass"
        monkeyatch_output = f"Found 1 classes with name MyClass in file file.py:\n\n- Search result 1:\n```{dummy_code}```\n"
        monkeypatch.setattr(
            sb,
            "search_class_in_file",
            lambda **kwargs: (
                monkeyatch_output,
                [SearchResult("file.py", 1, 2, "MyClass", None, dummy_code)],
                True,
            ),
        )
        result = sb.retrieve_class_context({("MyClass", "file.py")})
        expected_prefix = (
            "As additional context, here are the complete definitions of the classes "
            "around the more specific methods.\n"
        )
        assert result.startswith(expected_prefix)
        assert dummy_code in result

    def test_get_inherited_methods_no_override(self):
        sb = SearchBackend(project_path="dummy_project")
        # Setup indexes such that no inherited override exists.
        sb.class_relation_index = {"Child": []}
        sb.class_func_index = {"Parent": {"method": [("file.py", (10, 20))]}}
        # Call with a class having no parents.
        output, search_res, ok = sb._get_inherited_methods("Child", "method")
        assert output == ""
        assert search_res == []
        assert ok is False

    def test_get_inherited_methods_depth_break(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")

        # Set up the class relation hierarchy:
        # "A" directly inherits from "B" and "D". Then, "B" inherits from "E".
        # We'll mark that "D" overrides the method, and "E" also does but should be ignored.
        sb.class_relation_index = {"A": ["B", "D"], "B": ["E"], "D": [], "E": []}

        # Set up function index:
        # "B" does not override 'foo', "D" does, and "E" does too.
        sb.class_func_index = {
            "B": {},
            "D": {"foo": "dummy signature"},
            "E": {"foo": "dummy override"},
        }

        # Monkey-patch search_method_in_class so that when it is called,
        # it returns a dummy result for the override.
        monkeypatch.setattr(
            sb,
            "search_method_in_class",
            lambda super_call: (
                "dummy output",
                [SearchResult("/path/dummy.py", 1, 10, "D", "foo", "code snippet")],
                True,
            ),
        )

        output, search_res, ok = sb._get_inherited_methods("A", "foo")

        # We expect that only the override from "D" (depth 1) is processed.
        expected_snippet = "As additional context, this is an overriden instance of the method foo inside class D"
        assert expected_snippet in output
        assert len(search_res) == 1
        assert ok is True

    def test_get_inherited_methods_call_not_ok(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")

        # Set up a simple relation where "A" inherits from "B".
        sb.class_relation_index = {"A": ["B"]}
        # "B" overrides the method "foo".
        sb.class_func_index = {"B": {"foo": "dummy signature"}}

        # Monkey-patch search_method_in_class to simulate a failure (call_ok=False).
        monkeypatch.setattr(
            sb,
            "search_method_in_class",
            lambda super_call: (
                "dummy output",
                [SearchResult("/path/dummy.py", 1, 10, "B", "foo", "code snippet")],
                False,
            ),
        )

        output, search_res, ok = sb._get_inherited_methods("A", "foo")

        # Since call_ok is False, nothing should be appended to the final output.
        # Therefore, output should be empty, search_res should be empty, and ok False.
        assert output == ""
        assert search_res == []
        assert ok is False

    def test_get_inherited_methods_found(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        sb.class_relation_index = {"Child": ["Parent"]}
        sb.class_func_index = {"Parent": {"method": [("file.py", (10, 20))]}}
        dummy_sr = SearchResult("file.py", 10, 20, "Parent", "method", "dummy code")
        # In _get_inherited_methods, self.search_method_in_class is called with one argument (a dict)
        monkeypatch.setattr(
            sb,
            "search_method_in_class",
            lambda super_call: ("dummy output", [dummy_sr], True),
        )
        output, search_res, ok = sb._get_inherited_methods("Child", "method")
        assert "overriden instance" in output
        assert dummy_sr in search_res
        assert ok is True

    def test_get_bug_loc_snippets_new_with_method_and_class(
        self, tmp_path, monkeypatch
    ):
        # Create the directory and temporary file.
        temp_dir = tmp_path / "dummy_project"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / "temp_file.py"
        # Write multiple lines so that line indices 5 to 15 are valid.
        temp_file.write_text("\n".join([f"line {i}" for i in range(1, 21)]))
        # Use the absolute temporary directory as the project path.
        sb = SearchBackend(project_path=str(temp_dir))
        # Ensure parsed_files contains the temporary file.
        sb.parsed_files = [str(temp_file)]
        bug_loc = {
            "file": "temp_file.py",  # should match the end of the temp file path
            "method": "do_work",
            "class": "Worker",
            "intended_behavior": "Expected behavior description",
        }

        # Monkey-patch search_method_in_class to return a dummy search result.
        dummy_method_sr = SearchResult(
            str(temp_file), 5, 15, "Worker", "do_work", "method code"
        )
        monkeypatch.setattr(
            sb,
            "search_method_in_class",
            lambda m, c: ("method output", [dummy_method_sr], True),
        )

        # Monkey-patch search_class_in_file to return another dummy result.
        dummy_class_sr = SearchResult(
            str(temp_file), 1, 20, "Worker", None, "class code"
        )
        monkeypatch.setattr(
            sb,
            "search_class_in_file",
            lambda c, f: ("class output", [dummy_class_sr], True),
        )

        # Fallback functions: make them return no result.
        noop = lambda *args, **kwargs: ("", [], False)
        monkeyatch_noop = noop
        monkeypatch.setattr(sb, "search_method", monkeyatch_noop)
        monkeypatch.setattr(sb, "get_class_full_snippet", monkeyatch_noop)
        monkeypatch.setattr(sb, "search_method_in_file", monkeyatch_noop)
        monkeypatch.setattr(sb, "get_file_content", monkeyatch_noop)

        # Monkey-patch get_code_snippets so it returns a dummy snippet without reading the file.
        monkeypatch.setattr(
            "app.search.search_utils.get_code_snippets",
            lambda file_path, start, end, with_lineno=True: f"dummy snippet for {file_path} {start}-{end}",
        )

        bug_locs = sb.get_bug_loc_snippets_new(bug_loc)

        # We expect at least two BugLocations: one from the method branch and one from the class context.
        assert (
            len(bug_locs) >= 2
        ), f"Expected at least 2 BugLocations, got {len(bug_locs)}"
        for loc in bug_locs:
            assert loc.abs_file_path == str(temp_file)
            # Assuming apputils.to_relative_path returns the file name as relative path.
            assert loc.rel_file_path == "temp_file.py"
            # For BugLocations from search_method_in_class, intended_behavior should match the input.
            # For those from search_class_in_file (class context), it is overwritten.
            if loc.method_name is not None:
                expected_intended = "Expected behavior description"
            else:
                expected_intended = (
                    "This class provides additional context to the issue."
                )
            assert (
                loc.intended_behavior == expected_intended
            ), f"Expected '{expected_intended}', got '{loc.intended_behavior}'"

    def test_bug_loc_split_exact_two_fragments(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        bug_location_dict = {
            "file": "dummy.py",
            "method": "ClassA.methodA",
            "class": "",
            "intended_behavior": "do something",
        }

        # Patch get_code_snippets to avoid FileNotFoundError.
        monkeypatch.setattr(
            "app.search.search_utils.get_code_snippets",
            lambda file_full_path, start, end, with_lineno=True: "dummy snippet",
        )

        # Simulate a successful lookup in a method within a class.
        monkeypatch.setattr(
            sb,
            "search_method_in_class",
            lambda method, cls: (
                "output from search_method_in_class",
                [SearchResult("dummy.py", 10, 20, "ClassA", "methodA", "code snippet")],
                True,
            ),
        )

        # Simulate inherited method lookup.
        monkeypatch.setattr(
            sb,
            "_get_inherited_methods",
            lambda cls, method: (
                "inherited output",
                [
                    SearchResult(
                        "dummy.py", 5, 9, "BaseClass", "methodA", "base snippet"
                    )
                ],
                True,
            ),
        )

        # Simulate additional class context lookup.
        monkeypatch.setattr(
            sb,
            "search_class_in_file",
            lambda cls, file: (
                "class output",
                [SearchResult("dummy.py", 1, 30, "ClassA", None, "class snippet")],
                True,
            ),
        )

        bug_locs = sb.get_bug_loc_snippets_new(bug_location_dict)
        # Expect bug locations from search_method_in_class plus inherited and class context results.
        assert bug_locs
        # Optionally, inspect attributes of bug_locs here.

    def test_bug_loc_split_too_many_fragments(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        bug_location_dict = {
            "file": "dummy.py",
            "method": "A.B.C",  # Too many fragments should trigger warning and fallback.
            "class": "",
            "intended_behavior": "do something else",
        }

        # Patch get_code_snippets to avoid FileNotFoundError.
        monkeypatch.setattr(
            "app.search.search_utils.get_code_snippets",
            lambda file_full_path, start, end, with_lineno=True: "dummy snippet",
        )

        # Fallback search using search_method_in_file.
        monkeypatch.setattr(
            sb,
            "search_method_in_file",
            lambda method, file: (
                "output from search_method_in_file",
                [SearchResult("dummy.py", 15, 25, None, "A.B.C", "fallback snippet")],
                True,
            ),
        )

        bug_locs = sb.get_bug_loc_snippets_new(bug_location_dict)
        assert bug_locs
        # Further assertions on bug_locs can be made here.

    def test_bug_loc_fallback_method_in_file(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        bug_location_dict = {
            "file": "dummy.py",
            "method": "nonexistent_method",
            "class": "NonexistentClass",
            "intended_behavior": "should fallback",
        }

        # Patch get_code_snippets to avoid FileNotFoundError.
        monkeypatch.setattr(
            "app.search.search_utils.get_code_snippets",
            lambda file_full_path, start, end, with_lineno=True: "dummy snippet",
        )

        # Force search_method_in_class to fail.
        monkeypatch.setattr(
            sb, "search_method_in_class", lambda method, cls: ("", [], False)
        )

        # Fallback to search_method_in_file.
        monkeypatch.setattr(
            sb,
            "search_method_in_file",
            lambda method, file: (
                "output from search_method_in_file",
                [
                    SearchResult(
                        "dummy.py",
                        30,
                        40,
                        None,
                        "nonexistent_method",
                        "fallback snippet",
                    )
                ],
                True,
            ),
        )

        bug_locs = sb.get_bug_loc_snippets_new(bug_location_dict)
        assert bug_locs
        # Further assertions can be added here.

    def test_bug_loc_skip_invalid_results(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        bug_location_dict = {
            "file": "dummy.py",
            "method": "methodX",
            "class": "ClassX",
            "intended_behavior": "behave correctly",
        }

        # Patch get_code_snippets to avoid FileNotFoundError.
        monkeypatch.setattr(
            "app.search.search_utils.get_code_snippets",
            lambda file_full_path, start, end, with_lineno=True: "dummy snippet",
        )

        # Simulate search_method_in_class returning one invalid and one valid SearchResult.
        def fake_search_method_in_class(method, cls):
            return (
                "output",
                [
                    SearchResult(
                        "dummy.py", None, None, "ClassX", "methodX", "invalid snippet"
                    ),
                    SearchResult(
                        "dummy.py", 100, 110, "ClassX", "methodX", "valid snippet"
                    ),
                ],
                True,
            )

        monkeypatch.setattr(sb, "search_method_in_class", fake_search_method_in_class)

        # Return empty results for inherited and class context lookups.
        monkeypatch.setattr(
            sb, "_get_inherited_methods", lambda cls, method: ("", [], True)
        )
        monkeypatch.setattr(
            sb, "search_class_in_file", lambda cls, file: ("", [], True)
        )

        bug_locs = sb.get_bug_loc_snippets_new(bug_location_dict)
        # Only the valid SearchResult (with proper start/end) should produce a BugLocation.
        assert len(bug_locs) == 1

    def test_bug_loc_no_valid_search(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        bug_location_dict = {
            "file": "nonexistent.py",
            "method": "nonexistent_method",
            "class": "NonexistentClass",
            "intended_behavior": "nothing",
        }

        # Patch get_code_snippets to avoid FileNotFoundError.
        monkeypatch.setattr(
            "app.search.search_utils.get_code_snippets",
            lambda file_full_path, start, end, with_lineno=True: "dummy snippet",
        )

        # Force all search functions to fail.
        monkeypatch.setattr(sb, "search_method_in_class", lambda m, c: ("", [], False))
        monkeypatch.setattr(sb, "search_method_in_file", lambda m, f: ("", [], False))
        monkeypatch.setattr(sb, "search_class_in_file", lambda c, f: ("", [], False))
        monkeypatch.setattr(sb, "get_class_full_snippet", lambda c: ("", [], False))
        monkeypatch.setattr(sb, "search_method", lambda m: ("", [], False))
        monkeypatch.setattr(sb, "get_file_content", lambda f: ("", [], False))

        bug_locs = sb.get_bug_loc_snippets_new(bug_location_dict)
        # Expect an empty list because all search functions failed.
        assert bug_locs == []

    def test_get_bug_loc_snippets_invalid_fields(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        bug_location_dict = {
            "file": "dummy.py",
            "method": "TestMethod",
            "class": "TestClass",
            "intended_behavior": "intended behavior",
        }

        # Record calls for _get_inherited_methods and search_class_in_file.
        inherited_calls = []

        def fake_get_inherited_methods(cls, method):
            inherited_calls.append(1)
            # Return one valid SearchResult from inherited lookup.
            return (
                "inherited output",
                [
                    SearchResult(
                        "dummy.py",
                        21,
                        29,
                        "TestClass",
                        "TestMethod",
                        "inherited snippet",
                    )
                ],
                True,
            )

        sb._get_inherited_methods = fake_get_inherited_methods

        class_context_calls = []

        def fake_search_class_in_file(cls, file):
            class_context_calls.append(1)
            # Return one valid SearchResult from class context lookup.
            return (
                "class output",
                [SearchResult("dummy.py", 1, 9, "TestClass", None, "class snippet")],
                True,
            )

        sb.search_class_in_file = fake_search_class_in_file

        # Prepare three SearchResults from search_method_in_class.
        # SR1: Invalid for inherited lookup (missing class_name).
        sr1 = SearchResult("dummy.py", 10, 20, None, "TestMethod", "snippet1")
        # SR2: Fully valid.
        sr2 = SearchResult("dummy.py", 30, 40, "TestClass", "TestMethod", "snippet2")
        # SR3: Invalid for BugLocation conversion (missing start).
        sr3 = SearchResult("dummy.py", None, 50, "TestClass", "TestMethod", "snippet3")

        def fake_search_method_in_class(method, cls):
            # Return the three results and indicate success.
            return ("output", [sr1, sr2, sr3], True)

        sb.search_method_in_class = fake_search_method_in_class

        # Ensure that fallback searches are not triggered.
        sb.search_method_in_file = lambda m, f: ("", [], False)
        sb.get_class_full_snippet = lambda cls: ("", [], False)
        sb.search_method = lambda m: ("", [], False)
        sb.get_file_content = lambda f: ("", [], False)

        # Patch get_code_snippets (used in BugLocation creation) to avoid file I/O.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            lambda fname, start, end, with_lineno=True: "dummy snippet",
        )

        bug_locs = sb.get_bug_loc_snippets_new(bug_location_dict)

        # Explanation:
        # - search_method_in_class returns [sr1, sr2, sr3].
        # - For each result:
        #     * SR1: Skipped for inherited/class-context (due to missing class_name).
        #     * SR2: Triggers inherited and class-context lookups (adding 2 SearchResults).
        #     * SR3: Triggers inherited and class-context lookups (adding 2 SearchResults), even though SR3 itself is later skipped.
        # - Final search_res becomes:
        #       [SR1, SR2, SR3, inherited from SR2, inherited from SR3, class-context from SR2, class-context from SR3]
        # - BugLocation creation filters out any result with missing start/end.
        #       SR1: valid (start=10, end=20)
        #       SR2: valid (start=30, end=40)
        #       SR3: skipped (start is None)
        #       Both inherited results: valid (start=21, end=29)
        #       Both class-context results: valid (start=1, end=9)
        # - Total expected BugLocations: 1 (SR1) + 1 (SR2) + 2 (inherited) + 2 (class-context) = 6.

        assert len(bug_locs) == 6, f"Expected 6 bug locations, got {len(bug_locs)}"
        # Check that the lookup functions were called for both SR2 and SR3.
        assert (
            len(inherited_calls) == 2
        ), f"Expected inherited lookup to be called twice, got {len(inherited_calls)}"
        assert (
            len(class_context_calls) == 2
        ), f"Expected class context lookup to be called twice, got {len(class_context_calls)}"

    def test_get_bug_loc_skips_invalid_class_context(self, monkeypatch):
        sb = SearchBackend(project_path="dummy_project")
        bug_location_dict = {
            "file": "dummy.py",
            "method": "TestMethod",
            "class": "TestClass",
            "intended_behavior": "intended behavior",
        }

        # Fake search_method_in_class returns one valid result.
        valid_sr = SearchResult(
            "dummy.py", 10, 20, "TestClass", "TestMethod", "snippet"
        )

        def fake_search_method_in_class(method, cls):
            return ("output", [valid_sr], True)

        sb.search_method_in_class = fake_search_method_in_class

        # Ensure _get_inherited_methods returns no additional results (to isolate class context branch).
        sb._get_inherited_methods = lambda cls, method: ("", [], True)

        # Patch search_class_in_file to return one invalid SearchResult (e.g. missing start).
        def fake_search_class_in_file(cls, file):
            # Return a result with missing start (invalid for BugLocation conversion).
            invalid_sr = SearchResult(
                "dummy.py", None, 30, "TestClass", None, "class snippet"
            )
            return ("class output", [invalid_sr], True)

        sb.search_class_in_file = fake_search_class_in_file

        # Disable fallback searches.
        sb.search_method_in_file = lambda m, f: ("", [], False)
        sb.get_class_full_snippet = lambda cls: ("", [], False)
        sb.search_method = lambda m: ("", [], False)
        sb.get_file_content = lambda f: ("", [], False)

        # Patch get_code_snippets to avoid file I/O.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            lambda fname, start, end, with_lineno=True: "dummy snippet",
        )

        bug_locs = sb.get_bug_loc_snippets_new(bug_location_dict)

        # Explanation:
        # - The valid SearchResult from fake_search_method_in_class is added to search_res and converted into a BugLocation.
        # - Its corresponding call to fake_search_class_in_file returns an invalid SearchResult (start is None).
        # - In the final loop over class_context_search_res, the invalid SearchResult is skipped.
        # - Hence, only one valid BugLocation (from the valid search_result) is returned.

        assert len(bug_locs) == 1, f"Expected 1 bug location, got {len(bug_locs)}"
