import os
import textwrap

from tempfile import TemporaryDirectory

from app.search.search_backend import LineRange, SearchBackend, SearchResult


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
    top_level_funcs = [("func", 4, 4)] #TODO: clarify behavior of top-level functions, should 'method' inside class A be included?
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
        sb.class_relation_index = {}  # even though originally a defaultdict, we reset for test simplicity.
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
            monkeypatch.setattr("app.search.search_backend.search_utils.find_python_files", fake_find_python_files)
            monkeypatch.setattr("app.search.search_backend.search_utils.parse_python_file", fake_parse_python_file)

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
        sb.class_func_index = {
            "MyClass": {"my_method": [(dummy_file, (10, 20))]}
        }
        
        # Set up the top-level function index.
        # Structure: {function_name: [(file_name, (start, end))]}
        sb.function_index = {
            "top_func": [(dummy_file, (30, 40))]
        }
        
        # Test case 1: Line inside a class method.
        # The line 15 should be within the method "my_method" of "MyClass".
        result = sb._file_line_to_class_and_func(dummy_file, 15)
        assert result == ("MyClass", "my_method"), f"Expected ('MyClass', 'my_method'), got {result}"
        
        # Test case 2: Line inside a top-level function.
        # The line 35 should be within the top-level function "top_func".
        result = sb._file_line_to_class_and_func(dummy_file, 35)
        assert result == (None, "top_func"), f"Expected (None, 'top_func'), got {result}"
        
        # Test case 3: Line not within any function.
        # The line 50 should not be captured by any index.
        result = sb._file_line_to_class_and_func(dummy_file, 50)
        assert result == (None, None), f"Expected (None, None), got {result}"

    def test_search_func_in_class(self, monkeypatch):        
        # Monkey-patch the search_utils.get_code_snippets function.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets", dummy_get_code_snippets
        )

        # Create a dummy SearchBackend instance.
        sb = SearchBackend(project_path="dummy_project")
        
        # Set up the class_func_index with a sample entry.
        dummy_file = "dummy.py"
        sb.class_func_index = {
            "TestClass": {
                "func": [(dummy_file, (10, 20))]
            }
        }
        
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
    
    def test_search_func_in_all_classes(self, monkeypatch):
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets", dummy_get_code_snippets
        )

        sb = SearchBackend(project_path="dummy_project")
        
        dummy_file1 = "/absolute/path/file1.py"
        dummy_file2 = "/absolute/path/file2.py"
        
        # Set up the class_func_index with function "common" in two classes.
        sb.class_func_index = {
            "ClassA": {
                "common": [(dummy_file1, (5, 15))]
            },
            "ClassB": {
                "common": [(dummy_file2, (25, 35))]
            },
        }

        # Populate class_index so that _search_func_in_all_classes iterates over the classes.
        sb.class_index = {
            "ClassA": [(dummy_file1, (0, 100))],
            "ClassB": [(dummy_file2, (0, 100))]
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
        sb.function_index = {
            "top_func": [(str(dummy_file), (30, 40))]
        }
        
        # Monkey-patch the get_code_snippets function.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            dummy_get_code_snippets
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

    def test_search_func_in_code_base(self, tmp_path, monkeypatch):
        from app.search.search_backend import SearchBackend, SearchResult

        # Create three temporary Python files.
        file1 = tmp_path / "file1.py"
        file1.write_text(textwrap.dedent("""\
            def top_func():
                pass
        """))
        file2 = tmp_path / "file2.py"
        file2.write_text(textwrap.dedent("""\
            class ClassX:
                def top_func(self):
                    pass
        """))
        file3 = tmp_path / "file3.py"
        file3.write_text(textwrap.dedent("""\
            class ClassY:
                def top_func(self):
                    pass
        """))
        
        # Create a SearchBackend instance with the temporary directory as project path.
        sb = SearchBackend(project_path=str(tmp_path))
        
        # Set up the top-level function index.
        sb.function_index = {
            "top_func": [(str(file1), (1, 3))]
        }
        # Set up the class-function index with the same function in two classes.
        sb.class_func_index = {
            "ClassX": {
                "top_func": [(str(file2), (2, 4))]
            },
            "ClassY": {
                "top_func": [(str(file3), (2, 4))]
            }
        }
        # Populate class_index so that _search_func_in_all_classes iterates over the classes.
        sb.class_index = {
            "ClassX": [(str(file2), (1, 5))],
            "ClassY": [(str(file3), (1, 5))]
        }
        
        # Monkey-patch get_code_snippets with our dummy.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            dummy_get_code_snippets
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
        assert set(candidates) == expected_candidates, f"Expected {expected_candidates}, got {candidates}"
        
        # Test 2: Find files ending with "BAR.py" (case-insensitive).
        candidates = sb._get_candidate_matched_py_files("BAR.py")
        expected_candidates = {"/abs/path/bar.PY"}
        assert set(candidates) == expected_candidates, f"Expected {expected_candidates}, got {candidates}"
        
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
    
    def test_get_class_full_snippet_found(self, monkeypatch):

        sb = SearchBackend(project_path="dummy_project")
        
        # Set up class_index with a sample class "A" and one occurrence.
        sb.class_index = {
            "A": [("/absolute/path/fileA.py", (1, 10))]
        }
        
        # Monkey-patch get_code_snippets so it returns a predictable dummy snippet.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_code_snippets",
            lambda file_path, start, end, with_lineno=True: f"dummy code snippet from {file_path} lines {start}-{end}"
        )
        
        # Monkey-patch SearchResult.to_tagged_str to return a predictable tagged string.
        monkeypatch.setattr(
            SearchResult, "to_tagged_str",
            lambda self, project_path: f"tagged snippet from {self.file_path} {self.start}-{self.end}"
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

    def test_search_class_found_single(self, monkeypatch):
        from app.search.search_backend import SearchBackend, SearchResult, RESULT_SHOW_LIMIT

        sb = SearchBackend(project_path="dummy_project")
        
        # Set up class_index with one occurrence of class "MyClass".
        sb.class_index = {
            "MyClass": [("/absolute/path/fileA.py", (1, 10))]
        }
        
        # Override get_class_signature to return a predictable signature.
        monkeypatch.setattr(
            "app.search.search_backend.search_utils.get_class_signature",
            lambda fname, cname: f"Signature for {cname} in {fname}"
        )
        
        # Override SearchResult.to_tagged_str to return a predictable tagged string.
        monkeypatch.setattr(
            SearchResult,
            "to_tagged_str",
            lambda self, project_path: f"Tagged signature for {self.file_path}"
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

