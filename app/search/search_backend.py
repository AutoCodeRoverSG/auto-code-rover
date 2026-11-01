from collections import defaultdict, namedtuple
from collections.abc import MutableMapping
from functools import cache
from pathlib import Path

import timeout_decorator
from loguru import logger

from app.data_structures import BugLocation, SearchResult
from app.search import search_utils
from app.utils import catch_all_and_log

LineRange = namedtuple("LineRange", ["start", "end"])

ClassIndexType = MutableMapping[str, list[tuple[str, LineRange]]]
ClassFuncIndexType = MutableMapping[
    str, MutableMapping[str, list[tuple[str, LineRange]]]
]
FuncIndexType = MutableMapping[str, list[tuple[str, LineRange]]]
ClassRelationIndexType = MutableMapping[str, list[str]]

RESULT_SHOW_LIMIT = 3


class SearchBackend:
    def __init__(self, project_path: str):
        self.project_path = project_path
        # list of all files ending with .py, which are likely not test files
        # These are all ABSOLUTE paths.
        self.parsed_files: list[str] = []

        # for file name in the indexes, assume they are absolute path
        # class name -> [(file_name, line_range)]
        self.class_index: ClassIndexType = {}

        # {class_name -> {func_name -> [(file_name, line_range)]}}
        # inner dict is a list, since we can have (1) overloading func names,
        # and (2) multiple classes with the same name, having the same method
        self.class_func_index: ClassFuncIndexType = {}

        # a partially complete map of all the subclass relations
        # {class_name -> [class_name]}
        self.class_relation_index: ClassRelationIndexType = defaultdict(list)

        # function name -> [(file_name, line_range)]
        self.function_index: FuncIndexType = {}
        self._build_index()

    def _build_index(self):
        """
        With all source code of the project, build two indexes:
            1. From class name to (source file, start line, end line)
            2. From function name to (source file, start line, end line)
        Since there can be two classes/functions with the same name, the mapping
        value is a list of tuples.
        This is for fast lookup whenever we receive a query.
        """
        self._update_indices(*self._build_python_index(self.project_path))

    def _update_indices(
        self,
        class_index: ClassIndexType,
        class_func_index: ClassFuncIndexType,
        function_index: FuncIndexType,
        class_relation_index: ClassRelationIndexType,
        parsed_files: list[str],
    ) -> None:
        self.class_index.update(class_index)
        self.class_func_index.update(class_func_index)
        self.function_index.update(function_index)
        self.class_relation_index.update(class_relation_index)
        self.parsed_files.extend(parsed_files)

    @classmethod
    @cache
    def _build_python_index(cls, project_path: str) -> tuple[
        ClassIndexType,
        ClassFuncIndexType,
        FuncIndexType,
        ClassRelationIndexType,
        list[str],
    ]:
        class_index: ClassIndexType = defaultdict(list)
        class_func_index: ClassFuncIndexType = defaultdict(lambda: defaultdict(list))
        function_index: FuncIndexType = defaultdict(list)
        class_relation_index: ClassRelationIndexType = defaultdict(list)

        py_files = search_utils.find_python_files(project_path)
        # holds the parsable subset of all py files
        parsed_py_files = []
        for py_file in py_files:
            file_info = search_utils.parse_python_file(py_file)
            if file_info is None:
                # parsing of this file failed
                continue
            parsed_py_files.append(py_file)
            # extract from file info, and form search index
            classes, class_to_funcs, top_level_funcs, class_relation_map = file_info

            # (1) build class index
            for c, start, end in classes:
                class_index[c].append((py_file, LineRange(start, end)))

            # (2) build class-function index
            for c, class_funcs in class_to_funcs.items():
                for f, start, end in class_funcs:
                    class_func_index[c][f].append((py_file, LineRange(start, end)))

            # (3) build (top-level) function index
            for f, start, end in top_level_funcs:
                function_index[f].append((py_file, LineRange(start, end)))

            # (4) build class-superclass index
            for (c, start, end), super_classes in class_relation_map.items():
                class_relation_index[c] = super_classes

        return (
            class_index,
            class_func_index,
            function_index,
            class_relation_index,
            parsed_py_files,
        )

    def _file_line_to_class_and_func(
        self, file_path: str, line_no: int
    ) -> tuple[str | None, str | None]:
        """
        Given a file path and a line number, return the class and function name.
        If the line is not inside a class or function, return None.
        """
        # check whether this line is inside a class
        for class_name in self.class_func_index:
            func_dict = self.class_func_index[class_name]
            for func_name, func_info in func_dict.items():
                for file_name, (start, end) in func_info:
                    if file_name == file_path and start <= line_no <= end:
                        return class_name, func_name

        # not in any class; check whether this line is inside a top-level function
        for func_name in self.function_index:
            for file_name, (start, end) in self.function_index[func_name]:
                if file_name == file_path and start <= line_no <= end:
                    return None, func_name

        # this file-line is not recorded in any of the indexes
        return None, None

    def _search_func_in_class(
        self, function_name: str, class_name: str
    ) -> list[SearchResult]:
        """
        Search for the function name in the class.
        Args:
            function_name (str): Name of the function.
            class_name (str): Name of the class.
        Returns:
            The list of code snippets searched.
        """
        result: list[SearchResult] = []
        if class_name not in self.class_func_index:
            return result
        if function_name not in self.class_func_index[class_name]:
            return result
        for fname, (start, end) in self.class_func_index[class_name][function_name]:
            func_code = search_utils.get_code_snippets(fname, start, end)
            res = SearchResult(fname, start, end, class_name, function_name, func_code)
            result.append(res)
        return result

    def _search_func_in_all_classes(self, function_name: str) -> list[SearchResult]:
        """
        Search for the function name in all classes.
        Args:
            function_name (str): Name of the function.
        Returns:
            The list of code snippets searched.
        """
        result: list[SearchResult] = []
        for class_name in self.class_index:
            res = self._search_func_in_class(function_name, class_name)
            result.extend(res)
        return result

    def _search_top_level_func(self, function_name: str) -> list[SearchResult]:
        """
        Search for top-level function name in the entire project.
        Args:
            function_name (str): Name of the function.
        Returns:
            The list of code snippets searched.
        """
        result: list[SearchResult] = []
        if function_name not in self.function_index:
            return result

        for fname, (start, end) in self.function_index[function_name]:
            func_code = search_utils.get_code_snippets(fname, start, end)
            res = SearchResult(fname, start, end, None, function_name, func_code)
            result.append(res)
        return result

    def _search_func_in_code_base(self, function_name: str) -> list[SearchResult]:
        """
        Search for this function, from both top-level and all class definitions.
        """
        result: list[SearchResult] = []  # list of (file_name, func_code)
        # (1) search in top level
        top_level_res = self._search_top_level_func(function_name)
        class_res = self._search_func_in_all_classes(function_name)
        result.extend(top_level_res)
        result.extend(class_res)
        return result

    def _get_candidate_matched_py_files(self, target_file_name: str):
        """
        Search for files in the project that may match target_file_name.

        Returns:
            - all matched files, in abs path.
        """
        parsed_files_lower = [f.lower() for f in self.parsed_files]
        parsed_files = zip(self.parsed_files, parsed_files_lower)
        target_lower = target_file_name.lower()

        candidates = []
        for orig_file, lower_file in parsed_files:
            if lower_file.endswith(target_lower):
                candidates.append(orig_file)
        return candidates

    ###############################
    ### Interfaces ################
    ###############################
    ## NOTE: SearchResult objects returned by search APIs are not used when
    ## communicating with model - they are mainly for our own use cases.
    ## Only the first `tool_result` returned value is what sent to the model.

    # not search API - for writing patch
    # if we are searching for only a class when writing patch, likely we do not have enough info
    # the result can be too long, so we just show the first two
    # TODO: what to do with this method? It's not a method exposed to the agent, but maybe we also
    # want to catch exceptions from it?
    @catch_all_and_log
    def get_class_full_snippet(
        self, class_name: str
    ) -> tuple[str, list[SearchResult], bool]:
        search_res: list[SearchResult] = []
        tool_result = f"Could not find class {class_name} in the codebase."

        if class_name not in self.class_index:
            return tool_result, search_res, False

        for fname, (start, end) in self.class_index[class_name]:
            code = search_utils.get_code_snippets(fname, start, end)
            res = SearchResult(fname, start, end, class_name, None, code)
            search_res.append(res)

        if not search_res:
            return tool_result, search_res, False

        # the good path
        # for all the searched result, append them and form the final result
        tool_result = f"Found {len(search_res)} classes with name {class_name} in the codebase:\n\n"

        if len(search_res) > 2:
            tool_result += "Too many results, showing full code for 2 of them:\n"

        final_search_res = search_res[:2]
        for idx, res in enumerate(final_search_res):
            res_str = res.to_tagged_str(self.project_path)
            tool_result += f"- Search result {idx + 1}:\n```\n{res_str}\n```"
        return tool_result, final_search_res, True

    @catch_all_and_log
    def search_class(self, class_name: str) -> tuple[str, list[SearchResult], bool]:
        """Search for a class in the codebase.

        Only the signature of the class is returned. The class signature
        includes class name, base classes, and signatures for all of its methods/properties.

        Args:
            class_name (string): Name of the class to search for.
        """
        # initialize them to error case
        search_res: list[SearchResult] = []
        tool_result = f"Could not find class {class_name} in the codebase."

        if class_name not in self.class_index:
            return tool_result, search_res, False

        for fname, (start, end) in self.class_index[class_name]:
            # there are some classes; we return their signatures
            code = search_utils.get_class_signature(fname, class_name)
            res = SearchResult(fname, start, end, class_name, None, code)
            search_res.append(res)

        if not search_res:
            # this should not happen, but just in case
            return tool_result, search_res, False

        # the good path
        # for all the searched result, append them and form the final result
        tool_result = f"Found {len(search_res)} classes with name {class_name} in the codebase:\n\n"
        if len(search_res) > RESULT_SHOW_LIMIT:
            tool_result += "They appeared in the following files:\n"
            tool_result += SearchResult.collapse_to_file_level(
                search_res, self.project_path
            )
        else:
            for idx, res in enumerate(search_res):
                res_str = res.to_tagged_str(self.project_path)
                tool_result += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        final_search_res = search_res[:RESULT_SHOW_LIMIT]
        return tool_result, final_search_res, True

    @catch_all_and_log
    def search_class_in_file(
        self, class_name, file_name: str
    ) -> tuple[str, list[SearchResult], bool]:
        """Search for a class in a given file.

        Returns the actual code of the entire class definition.

        Args:
            class_name (string): Name of the class to search for.
            file_name (string): The file to search in. Must be a valid python file name.
        """
        search_res: list[SearchResult] = []

        # (1) check whether we can get the file
        candidate_py_abs_paths = self._get_candidate_matched_py_files(file_name)
        if not candidate_py_abs_paths:
            tool_output = f"Could not find file {file_name} in the codebase."
            return tool_output, search_res, False

        # (2) search for this class in the entire code base (we do filtering later)
        if class_name not in self.class_index:
            tool_output = f"Could not find class {class_name} in the codebase."
            return tool_output, search_res, False

        # (3) class is there, check whether it exists in the file specified.
        for fname, (start, end) in self.class_index[class_name]:
            if fname in candidate_py_abs_paths:
                class_code = search_utils.get_code_snippets(fname, start, end)
                res = SearchResult(fname, start, end, class_name, None, class_code)
                search_res.append(res)

        if not search_res:
            tool_output = f"Could not find class {class_name} in file {file_name}."
            return tool_output, search_res, False

        # good path; we have result, now just form a response
        tool_output = f"Found {len(search_res)} classes with name {class_name} in file {file_name}:\n\n"
        for idx, res in enumerate(search_res):
            res_str = res.to_tagged_str(self.project_path)
            tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        return tool_output, search_res, True

    @catch_all_and_log
    def search_method_in_file(
        self, method_name: str, file_name: str
    ) -> tuple[str, list[SearchResult], bool]:
        """Search for a method in a given file.

        Returns the actual code of the method.

        Args:
            method_name (string): Name of the method to search for.
            file_name (string): The file to search in. Must be a valid python file name.
        """
        # (1) check whether we can get the file
        # supports both when file_name is relative to project root, and when
        # it is just a short name
        candidate_py_abs_paths = self._get_candidate_matched_py_files(file_name)
        # print(candidate_py_files)
        if not candidate_py_abs_paths:
            tool_output = f"Could not find file {file_name} in the codebase."
            return tool_output, [], False

        # (2) search for this method in the entire code base (we do filtering later)
        search_res: list[SearchResult] = self._search_func_in_code_base(method_name)
        if not search_res:
            tool_output = f"The method {method_name} does not appear in the codebase."
            return tool_output, [], False

        # (3) filter the search result => they need to be in one of the files!
        filtered_res: list[SearchResult] = [
            res for res in search_res if res.file_path in candidate_py_abs_paths
        ]

        # (4) done with search, now prepare result
        if not filtered_res:
            tool_output = (
                f"There is no method with name `{method_name}` in file {file_name}."
            )
            return tool_output, [], False

        tool_output = f"Found {len(filtered_res)} methods with name `{method_name}` in file {file_name}:\n\n"

        # when searching for a method in one file, it's rare that there are
        # many candidates, so we do not trim the result
        for idx, res in enumerate(filtered_res):
            res_str = res.to_tagged_str(self.project_path)
            tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        return tool_output, filtered_res, True

    @catch_all_and_log
    def search_method_in_class(
        self, method_name: str, class_name: str
    ) -> tuple[str, list[SearchResult], bool]:
        """Search for a method in a given class.

        Returns the actual code of the method.

        Args:
            method_name (string): Name of the method to search for.
            class_name (string): Consider only methods in this class.
        """
        if class_name not in self.class_index:
            tool_output = f"Could not find class {class_name} in the codebase."
            return tool_output, [], False

        # has this class, check its methods
        search_res: list[SearchResult] = self._search_func_in_class(
            method_name, class_name
        )
        if not search_res:
            tool_output = f"Could not find method {method_name} in class {class_name}`."
            return tool_output, [], False

        # found some methods, prepare the result
        tool_output = f"Found {len(search_res)} methods with name {method_name} in class {class_name}:\n\n"

        # There can be multiple classes defined in multiple files, which contain the same method
        # still trim the result, just in case
        if len(search_res) > RESULT_SHOW_LIMIT:
            tool_output += f"Too many results, showing full code for {RESULT_SHOW_LIMIT} of them, and the rest just file names:\n"
        first_five = search_res[:RESULT_SHOW_LIMIT]
        for idx, res in enumerate(first_five):
            res_str = res.to_tagged_str(self.project_path)
            tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
        # for the rest, collect the file names into a set
        if rest := search_res[RESULT_SHOW_LIMIT:]:
            tool_output += "Other results are in these files:\n"
            tool_output += SearchResult.collapse_to_file_level(rest, self.project_path)

        return tool_output, first_five, True

    @catch_all_and_log
    def search_method(self, method_name: str) -> tuple[str, list[SearchResult], bool]:
        """Search for a method in the entire codebase.

        Returns the actual code of the method.

        Args:
            method_name (string): Name of the method to search for.
        """
        search_res: list[SearchResult] = self._search_func_in_code_base(method_name)
        if not search_res:
            tool_output = f"Could not find method {method_name} in the codebase."
            return tool_output, [], False

        tool_output = f"Found {len(search_res)} methods with name {method_name} in the codebase:\n\n"

        if len(search_res) > RESULT_SHOW_LIMIT:
            tool_output += "They appeared in the following files:\n"
            tool_output += SearchResult.collapse_to_file_level(
                search_res, self.project_path
            )
        else:
            for idx, res in enumerate(search_res):
                res_str = res.to_tagged_str(self.project_path)
                tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"

        final_search_res = search_res[:RESULT_SHOW_LIMIT]
        return tool_output, final_search_res, True

    @catch_all_and_log
    @timeout_decorator.timeout(120)
    def search_code(self, code_str: str) -> tuple[str, list[SearchResult], bool]:
        """Search for a code snippet in the entire codebase.

        Returns the method that contains the code snippet, if it is found inside a method.
        Otherwise, returns the region of code surrounding it.

        Args:
            code_str (string): The code snippet to search for.
        """
        # attempt to search for this code string in all py files
        search_res: list[SearchResult] = []
        for file_path in self.parsed_files:
            searched_line_and_code: list[tuple[int, str]] = (
                search_utils.get_code_region_containing_code(file_path, code_str)
            )
            if not searched_line_and_code:
                continue
            for searched in searched_line_and_code:
                line_no, code_region = searched
                # from line_no, check which function and class we are in
                class_name, func_name = self._file_line_to_class_and_func(
                    file_path, line_no
                )
                res = SearchResult(
                    file_path, line_no, line_no, class_name, func_name, code_region
                )
                search_res.append(res)

        if not search_res:
            tool_output = f"Could not find code {code_str} in the codebase."
            return tool_output, [], False

        # good path
        tool_output = f"Found {len(search_res)} snippets containing `{code_str}` in the codebase:\n\n"

        if len(search_res) > RESULT_SHOW_LIMIT:
            tool_output += "They appeared in the following files:\n"
            tool_output += SearchResult.collapse_to_file_level(
                search_res, self.project_path
            )
        else:
            for idx, res in enumerate(search_res):
                res_str = res.to_tagged_str(self.project_path)
                tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"

        final_search_res = search_res[:RESULT_SHOW_LIMIT]
        return tool_output, final_search_res, True

    @catch_all_and_log
    def search_code_in_file(
        self, code_str: str, file_name: str
    ) -> tuple[str, list[SearchResult], bool]:
        """Search for a code snippet in a given file file.

        Returns the entire method that contains the code snippet.

        Args:
            code_str (string): The code snippet to search for.
            file_name (string): The file to search in. Must be a valid python file name in the project.
        """
        code_str = code_str.removesuffix(")")

        candidate_py_files = [f for f in self.parsed_files if f.endswith(file_name)]
        if not candidate_py_files:
            tool_output = f"Could not find file {file_name} in the codebase."
            return tool_output, [], False

        # start searching for code in the filtered files
        search_res: list[SearchResult] = []
        for file_path in candidate_py_files:
            searched_line_and_code: list[tuple[int, str]] = (
                search_utils.get_code_region_containing_code(file_path, code_str)
            )
            if not searched_line_and_code:
                continue
            for searched in searched_line_and_code:
                line_no, code_region = searched
                # from line_no, check which function and class we are in
                class_name, func_name = self._file_line_to_class_and_func(
                    file_path, line_no
                )
                res = SearchResult(
                    file_path, line_no, line_no, class_name, func_name, code_region
                )
                search_res.append(res)

        if not search_res:
            tool_output = f"Could not find code {code_str} in file {file_name}."
            return tool_output, [], False

        # good path
        # There can be a lot of results, from multiple files.
        tool_output = f"Found {len(search_res)} snippets with code {code_str} in file {file_name}:\n\n"
        if len(search_res) > RESULT_SHOW_LIMIT:
            tool_output += "They appeared in the following methods:\n"
            tool_output += SearchResult.collapse_to_method_level(
                search_res, self.project_path
            )
        else:
            for idx, res in enumerate(search_res):
                res_str = res.to_tagged_str(self.project_path)
                tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"

        final_search_res = search_res[:RESULT_SHOW_LIMIT]
        return tool_output, final_search_res, True

    @catch_all_and_log
    def get_code_around_line(
        self, file_name: str, line_no_str: str, window_size_str: str
    ) -> tuple[str, list[SearchResult], bool]:
        """
        Get the region of code around line `line_no` in the file `file_name`.

        Args:
            file_name (str): The file name.
            line_no_str (str): The line number. (1-based)
            window_size_str (str): The number of lines before and after the line number.
        """
        # we get argument as string
        line_no = int(line_no_str)
        window_size = int(window_size_str)

        # (1) check whether we can get the file
        candidate_py_abs_paths = self._get_candidate_matched_py_files(file_name)
        if not candidate_py_abs_paths:
            tool_output = f"Could not find file {file_name} in the codebase."
            return tool_output, [], False

        # (2) make a SearchResult for each file
        # region search result is what we will turn into the response to the model
        region_search_results: list[SearchResult] = []
        # func_search_results is what we keep for record
        func_search_results: list[SearchResult] = []

        for file_path in candidate_py_abs_paths:
            snippet = search_utils.get_code_region_around_line(
                file_path, line_no, window_size
            )
            if snippet is None:
                continue
            class_name, func_name = self._file_line_to_class_and_func(
                file_path, line_no
            )
            # get the surrounding functions, since our instrumentation is on function level
            if func_name is not None and class_name is not None:
                _, curr_func_results, _ = self.search_method_in_class(
                    func_name, class_name
                )
            elif func_name is not None:
                _, curr_func_results, _ = self.search_method(func_name)
            else:
                curr_func_results = []
            func_search_results.extend(curr_func_results)

            start_lineno = line_no - window_size
            end_lineno = line_no + window_size
            res = SearchResult(
                file_path, start_lineno, end_lineno, class_name, func_name, snippet
            )
            region_search_results.append(res)

        if not region_search_results:
            tool_output = f"{line_no} is invalid in file {file_name}."
            return tool_output, [], False

        # good path
        tool_output = f"Found {len(region_search_results)} code snippets around line {line_no}:\n\n"
        for idx, res in enumerate(region_search_results):
            res_str = res.to_tagged_str(self.project_path)
            tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"

        # NOTE: returning functions in search results, since they will be instrumented later
        return tool_output, func_search_results, True

    @catch_all_and_log
    def get_file_content(self, file_name: str) -> tuple[str, list[SearchResult], bool]:
        """Get actual content of the entire file.
        Mainly used for retrieving actual code snippets at selected bug locations.

        Args:
            - file_name: relevant path to the file.
        """
        # check whether we can get the file
        candidate_py_files = [f for f in self.parsed_files if f.endswith(file_name)]
        if not candidate_py_files:
            tool_output = f"Could not find file {file_name} in the codebase."
            return tool_output, [], False

        # NOTE: sometimes there can be multiple files.
        # To make the execution safe, we just take the first one

        file_path = candidate_py_files[0]
        file_content = Path(file_path).read_text()

        file_length = len(file_content.splitlines())

        search_res = [SearchResult(file_path, 1, file_length, None, None, file_content)]

        tool_output = (
            f"Found file {file_name} in the codebase:\n\n```\n{file_content}\n```\n"
        )
        tool_output = f"<file>{file_name}</file> <code>{file_content}</code>"
        return tool_output, search_res, True

    def retrieve_class_context(
        self, class_and_files: set[tuple[str, str]]
    ) -> str | None:
        """
        Args:
            - set of classes to retrieve as additional context.
            Each element is a tuple of (class_name, file_name).
        Returns:
            - A string containing definitions of all classes.
        """
        result_prefix = (
            "As additional context, here are the complete definitions of the classes "
            "around the more specific methods.\n"
        )
        result = ""

        for class_name, file_name in class_and_files:
            kwargs = {"class_name": class_name, "file_name": file_name}
            code, _, search_ok = self.search_class_in_file(**kwargs)
            if search_ok:
                result += f"\n\n{code}\n\n"

        if result:
            # some class definitions could be retrieved
            return result_prefix + result
        else:
            return None

    def _get_inherited_methods(self, class_name: str, method_name: str):
        """
        Given a method in a class, find its inherited classes in the parent class.
        Should eventually return whatever "search_method_in_class" returns.
        """
        class_queue: list[tuple[str, int]] = list(
            map(lambda n: (n, 1), self.class_relation_index[class_name])
        )
        super_calls: list[dict[str, str]] = []
        found_at_depth = -1
        while class_queue:
            (ancestor_name, depth) = class_queue.pop(0)
            if found_at_depth != -1 and depth > found_at_depth:
                break
            functions = self.class_func_index.get(ancestor_name, dict())
            if method_name in functions:
                found_at_depth = depth
                super_calls.append(
                    {"class_name": ancestor_name, "method_name": method_name}
                )
            else:
                for great_ancestor in self.class_relation_index.get(
                    ancestor_name, list()
                ):
                    class_queue.append((great_ancestor, depth + 1))

        final_output = ""
        final_search_res: list[SearchResult] = []

        if super_calls:
            for super_call in super_calls:
                logger.debug(
                    f"Found override of {super_call['method_name']} in {super_call['class_name']}"
                )

                output, search_res, call_ok = self.search_method_in_class(super_call)

                if not call_ok:
                    continue

                final_output += f"As additional context, this is an overriden instance of the method {method_name} inside class {super_call['class_name']}\n\n{output}\n\n"

                final_search_res.extend(search_res)

        return final_output, final_search_res, bool(final_output)

    def get_bug_loc_snippets_new(self, bug_location_dict: dict[str, str]):
        """
        Since this function is probably buggy, rewrite it.
        """
        # these are just what the model has returned us, so they may be wrong
        tmp_file_name = bug_location_dict.get("file", "")
        tmp_method_name = bug_location_dict.get("method", "")
        tmp_class_name = bug_location_dict.get("class", "")

        intended_behavior = bug_location_dict.get("intended_behavior", "")

        # (1) sometimes model can write class_name and method_name together in the
        # format Class.method

        tmp_class_name, tmp_method_name = self._split_class_method_name(
            tmp_class_name, tmp_method_name
        )

        # we require at least the file_name to be given
        assert (
            tmp_method_name or tmp_class_name or tmp_file_name
        ), f"Invalid bug location returned from model: {bug_location_dict}"

        # (2) start searching for this location in the codebase

        call_ok = False
        search_res: list[SearchResult] = []

        class_context_search_res: list[SearchResult] = []

        # (2.1) search for the method in the class
        # NOTE: make sure all search_res below contains a valid unit of code,
        # such as method/class/file. Also, the search_res should contain correct
        # line numbers for this code unit.
        # Due to legacy reasons, search_res returned by some functions do not
        # satisfy the requirement above, so DO NOT use those functions here.

        if tmp_method_name and tmp_class_name:
            search_res, class_context_search_res, call_ok = (
                self._search_method_in_class_with_context(
                    tmp_method_name, tmp_class_name
                )
            )

        if not call_ok:
            search_res, call_ok = self._search_fallback_locations(
                tmp_file_name, tmp_method_name, tmp_class_name
            )

        if not call_ok:
            # cannot find any location!!
            return []

        # we have some SearchResults => turn these into BugLocations
        final_bug_locs = self._search_results_to_bug_locs(
            search_res, intended_behavior
        )

        # deal with additional class context search results
        final_bug_locs.extend(
            self._search_results_to_bug_locs(
                class_context_search_res,
                "This class provides additional context to the issue.",
            )
        )

        return final_bug_locs

    @staticmethod
    def _split_class_method_name(
        tmp_class_name: str, tmp_method_name: str
    ) -> tuple[str, str]:
        """
        Sometimes model can write class_name and method_name together in the
        format Class.method. Split them if possible.
        """
        if tmp_class_name or not tmp_method_name or "." not in tmp_method_name:
            return tmp_class_name, tmp_method_name

        fragments = tmp_method_name.split(".")
        if len(fragments) != 2:
            logger.warning(
                "Too many fragments. Examine the method name: {}", tmp_method_name
            )
            return tmp_class_name, tmp_method_name

        tmp_class_name, tmp_method_name = fragments
        logger.warning("Successfully split {} and {}", tmp_class_name, tmp_method_name)
        return tmp_class_name, tmp_method_name

    def _search_method_in_class_with_context(
        self, method_name: str, class_name: str
    ) -> tuple[list[SearchResult], list[SearchResult], bool]:
        """
        Search for a method in a class. When found, also collect the inherited
        parent class methods, and the entire class as context.
        Returns (search_res, class_context_search_res, call_ok).
        """
        class_context_search_res: list[SearchResult] = []

        _, curr_search_res, call_ok = self.search_method_in_class(
            method_name, class_name
        )
        search_res: list[SearchResult] = list(curr_search_res)
        if not call_ok:
            return search_res, class_context_search_res, call_ok

        # when the location is decided to be method in a class, we also
        # obtain (1) the entire class as location, and (2) the inherited
        # parent class methods as location.
        res: SearchResult
        for res in curr_search_res:
            if not self._has_class_func_and_file(res):
                continue

            _, inherited_search_res, _ = self._get_inherited_methods(
                res.class_name, res.func_name
            )
            search_res.extend(inherited_search_res)

            # this kind of class is special, they just serve as a context,
            # so the 'intended_behavior' field content does not apply for them.
            _, class_search_res, _ = self.search_class_in_file(
                res.class_name, res.file_path
            )
            class_context_search_res.extend(class_search_res)

        return search_res, class_context_search_res, call_ok

    @staticmethod
    def _has_class_func_and_file(res: SearchResult) -> bool:
        return (
            res.class_name is not None
            and res.func_name is not None
            and res.file_path is not None
        )

    def _search_fallback_locations(
        self, tmp_file_name: str, tmp_method_name: str, tmp_class_name: str
    ) -> tuple[list[SearchResult], bool]:
        """
        Try progressively less specific searches until one succeeds.
        Returns (search_res, call_ok).
        """
        attempts = [
            (
                bool(tmp_method_name and tmp_file_name),
                lambda: self.search_method_in_file(tmp_method_name, tmp_file_name),
            ),
            (
                bool(tmp_class_name and tmp_file_name),
                lambda: self.search_class_in_file(tmp_class_name, tmp_file_name),
            ),
            (
                bool(tmp_class_name),
                lambda: self.get_class_full_snippet(tmp_class_name),
            ),
            (
                bool(tmp_method_name),
                lambda: self.search_method(tmp_method_name),
            ),
            (
                bool(tmp_file_name),
                lambda: self.get_file_content(tmp_file_name),
            ),
        ]

        search_res: list[SearchResult] = []
        for should_try, search_func in attempts:
            if not should_try:
                continue
            _, search_res, call_ok = search_func()
            if call_ok:
                return search_res, True

        return search_res, False

    def _search_results_to_bug_locs(
        self, search_res: list[SearchResult], intended_behavior: str
    ) -> list[BugLocation]:
        bug_locs: list[BugLocation] = []
        for res in search_res:
            if res.start is None or res.end is None:
                continue
            bug_locs.append(BugLocation(res, self.project_path, intended_behavior))
        return bug_locs


if __name__ == "__main__":
    pass
    ## Test parsing of bug locations
    # backend = SearchBackend("/media/media0/yuntong/SWE-bench/testbed/django__django/setup_django__django__3.0")
    # bug_locations = [
    #     {
    #         "file": "django/conf/global_settings.py",
    #         "class": "",
    #         "method": ""
    #     },
    #     {
    #         "file": "django/core/files/storage.py",
    #         "class": "FileSystemStorage",
    #         "method": "__init__"
    #     },
    #     {
    #         "file": "django/core/files/storage.py",
    #         "class": "FileSystemStorage",
    #         "method": "_save"
    #     },
    #     {
    #         "file": "tests/file_storage/tests.py",
    #         "class": "",
    #         "method": ""
    #     }
    # ]
    # for bug_location in bug_locations:
    #     print(backend.get_bug_loc_snippets(bug_location))

    ## Test class inheritance index
    # backend = SearchBackend(
    #     "/media/media0/yuntong/SWE-bench/testbed/django__django/setup_django__django__4.0"
    # )
    # loc = {
    #     "file": "django/db/models/fields/__init__.py",
    #     "class": "AutoFieldMeta",
    #     "method": "__subclasscheck__",
    # }
    # code = backend.get_bug_loc_snippets(loc)
    # print(code)

    # backend = SearchBackend("/media/media0/yuntong/SWE-bench/testbed/django__django/setup_django__django__3.0")

    # locs = [
    #     {
    #         "file": "django/utils/autoreload.py",
    #         "class": "StatReloader",
    #         "method": "snapshot_files",
    #         "intended_behavior": "The snapshot_files method should take a snapshot of the watched files and their modification times without encountering errors. Specifically, it should handle any file paths that contain unexpected null bytes gracefully, possibly by skipping such paths or logging a warning."
    #     },
    #     {
    #         "file": "django/utils/autoreload.py",
    #         "class": "StatReloader",
    #         "method": "watched_files",
    #         "intended_behavior": "The watched_files method should yield all files that need to be watched for changes without encountering errors. It should ensure that any file paths containing unexpected null bytes are handled gracefully, possibly by skipping such paths or logging a warning."
    #     },
    #     {
    #         "file": "django/utils/autoreload.py",
    #         "class": "StatReloader",
    #         "method": "run_loop",
    #         "intended_behavior": "The run_loop method should run the reloader loop, checking for file changes at regular intervals without encountering errors. It should ensure that any file paths containing unexpected null bytes are handled gracefully, possibly by skipping such paths or logging a warning."
    #     }
    # ]

    # for loc in locs:
    #     print(backend.get_bug_loc_snippets(loc))

    # backend = SearchBackend(
    #     "/media/media0/yuntong/SWE-bench/testbed/astropy__astropy/setup_astropy__astropy__1.3"
    # )
    # locs = [
    #     {
    #         "file": "astropy/wcs/wcs.py",
    #         "class": "WCS",
    #         "method": "_array_converter",
    #         "intended_behavior": "The _array_converter method should handle empty input arrays gracefully and return empty arrays without raising an error. This ensures that when methods like wcs_pix2world are called with empty lists, they return empty lists/arrays instead of raising an InconsistentAxisTypesError.",
    #     },
    #     {
    #         "file": "astropy/wcs/wcs.py",
    #         "class": "WCS",
    #         "method": "wcs_pix2world",
    #         "intended_behavior": "The wcs_pix2world method should utilize the modified _array_converter method to ensure that when it is called with empty lists, it returns empty lists/arrays instead of raising an error. This preserves the existing functionality while handling edge cases of empty inputs.",
    #     },
    # ]

    # for loc in locs:
    #     print(backend.get_bug_loc_snippets(loc))
