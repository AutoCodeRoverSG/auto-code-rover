import json
from pathlib import Path

import pytest

from app.search.search_manage import SearchManager
from app.search.search_backend import SearchResult, RESULT_SHOW_LIMIT
from app.data_structures import BugLocation, MessageThread
from app.task import Task
from app import config
from app.agents import agent_search, agent_proxy


# --- Dummy helper classes ---
class DummyMessageThread(MessageThread):
    def __init__(self):
        # minimal initialization
        pass

    def save_to_file(self, file_path):
        # simulate saving to file; write dummy content
        Path(file_path).write_text("dummy message thread content")


class DummyTask(Task):
    def __init__(self):
        # Provide necessary dummy state
        self._project_path = "dummy_project"

    def get_issue_statement(self):
        return "dummy issue statement"

    # Implement abstract methods with dummy behavior.
    def reset_project(self):
        pass

    def setup_project(self):
        pass

    def validate(self):
        pass

    @property
    def project_path(self):
        return self._project_path

# --- Test class for SearchManager ---
class TestSearchManager:
    def test_start_new_tool_call_layer_and_add(self, tmp_path):
        # Test start_new_tool_call_layer, add_tool_call_to_curr_layer, and dumping to file.
        output_dir = tmp_path / "output"
        sm = SearchManager(project_path="dummy_project", output_dir=str(output_dir))
        # Initially, tool_call_layers should be empty.
        assert sm.tool_call_layers == []
        sm.start_new_tool_call_layer()
        assert len(sm.tool_call_layers) == 1
        # Add a tool call and check.
        sm.add_tool_call_to_curr_layer("dummy_func", {"arg": "val"}, True)
        assert sm.tool_call_layers[-1] == [
            {"func_name": "dummy_func", "arguments": {"arg": "val"}, "call_ok": True}
        ]
        # Dump to file and verify file creation.
        sm.dump_tool_call_layers_to_file()
        dump_file = Path(sm.output_dir) / "tool_call_layers.json"
        assert dump_file.exists()
        content = json.loads(dump_file.read_text())
        assert isinstance(content, list)

    def test_search_iterative_success(self, monkeypatch, tmp_path):
        # Set conv_round_limit to 1 for testing.
        monkeypatch.setattr(config, "conv_round_limit", 1)
        # Create a temporary output directory.
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        sm = SearchManager(project_path="dummy_project", output_dir=str(output_dir))
        dummy_task = DummyTask()
        sbfl_result = "dummy sbfl result"
        reproducer_result = "dummy reproducer result"

        # Patch search_utils.get_code_snippets to avoid file I/O error.
        monkeypatch.setattr(
            "app.search.search_utils.get_code_snippets",
            lambda file, start, end, with_lineno=True: "dummy snippet"
        )

        # --- Dummy generator for agent_search ---
        def dummy_generator(issue, sbfl, reproducer):
            dummy_thread = DummyMessageThread()
            # First send returns a dummy agent search response and a dummy message thread.
            received = yield ("dummy agent search response", dummy_thread)
        monkeypatch.setattr(
            agent_search, "generator", 
            lambda issue, sbfl, reproducer: dummy_generator(issue, sbfl, reproducer)
        )

        # --- Dummy agent_proxy.run_with_retries ---
        # Return a valid JSON string with bug_locations and no API_calls.
        bug_loc_dict = {
            "file": "dummy.py",
            "method": "test_method",
            "class": "Dummy",
            "intended_behavior": "beh"
        }
        selected_apis = json.dumps({"API_calls": [], "bug_locations": [bug_loc_dict]})
        monkeypatch.setattr(agent_proxy, "run_with_retries", lambda resp: (selected_apis, []))

        # --- Patch backend.get_bug_loc_snippets_new ---
        dummy_sr = SearchResult("dummy.py", 1, 2, "Dummy", "test_method", "dummy code")
        dummy_bug_loc = BugLocation(dummy_sr, "dummy_project", "beh")
        monkeypatch.setattr(sm.backend, "get_bug_loc_snippets_new", lambda loc: [dummy_bug_loc])

        # Prevent actual printing.
        monkeypatch.setattr("app.search.search_manage.print_banner", lambda msg: None)
        monkeypatch.setattr("app.search.search_manage.print_acr", lambda msg, title: None)

        bug_locations, msg_thread = sm.search_iterative(dummy_task, sbfl_result, reproducer_result, None)
        # Validate that one dummy BugLocation is returned and the message thread is our dummy.
        assert bug_locations == [dummy_bug_loc]
        assert isinstance(msg_thread, DummyMessageThread)

    def test_search_iterative_no_valid_calls(self, monkeypatch, tmp_path):
        # Test a branch where no valid API call returns a location.
        monkeyatch_noop = lambda *args, **kwargs: ("", [], False)
        monkeypatch.setattr(config, "conv_round_limit", 1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        sm = SearchManager(project_path="dummy_project", output_dir=str(output_dir))
        dummy_task = DummyTask()
        sbfl_result = "dummy sbfl result"
        reproducer_result = "dummy reproducer result"

        # Dummy generator yielding a dummy response.
        def dummy_generator(issue, sbfl, reproducer):
            dummy_thread = DummyMessageThread()
            received = yield ("dummy agent search response", dummy_thread)
        monkeypatch.setattr(
            agent_search, "generator",
            lambda issue, sbfl, reproducer: dummy_generator(issue, sbfl, reproducer)
        )
        # Force agent_proxy.run_with_retries to return None for selected_apis.
        monkeypatch.setattr(agent_proxy, "run_with_retries", lambda resp: (None, []))

        # Force backend fallback functions to return no result.
        monkeypatch.setattr(sm.backend, "search_method_in_class", monkeyatch_noop)
        monkeyatch_noop = lambda *args, **kwargs: ("", [], False)
        monkeypatch.setattr(sm.backend, "search_method", monkeyatch_noop)
        monkeyatch_noop = lambda *args, **kwargs: ("", [], False)
        monkeypatch.setattr(sm.backend, "search_class_in_file", monkeyatch_noop)
        monkeypatch.setattr(sm.backend, "get_class_full_snippet", monkeyatch_noop)
        monkeyatch_noop = lambda *args, **kwargs: ("", [], False)
        monkeypatch.setattr(sm.backend, "get_file_content", monkeyatch_noop)

        monkeypatch.setattr("app.search.search_manage.print_banner", lambda msg: None)
        monkeypatch.setattr("app.search.search_manage.print_acr", lambda msg, title: None)

        bug_locations, msg_thread = sm.search_iterative(dummy_task, sbfl_result, reproducer_result, None)
        # Expect empty bug location list because no valid result was found.
        assert bug_locations == []
        assert isinstance(msg_thread, DummyMessageThread)

    def test_search_iterative_with_api_calls(self, monkeypatch, tmp_path):
        """
        Test the branch in search_iterative where a valid API call is returned.
        This test simulates a response with a non-empty API_calls list (and empty bug_locations),
        and ensures that the API call is processed (by invoking a dummy backend function)
        and recorded in the tool call layers.
        """
        monkeypatch.setattr(config, "conv_round_limit", 1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        sm = SearchManager(project_path="dummy_project", output_dir=str(output_dir))
        dummy_task = DummyTask()
        sbfl_result = "dummy sbfl result"
        reproducer_result = "dummy reproducer result"

        # Define dummy_generator for agent_search.
        def dummy_generator(issue, sbfl, reproducer):
            dummy_thread = DummyMessageThread()
            received = yield ("dummy agent search response", dummy_thread)
        monkeypatch.setattr(
            agent_search, "generator",
            lambda issue, sbfl, reproducer: dummy_generator(issue, sbfl, reproducer)
        )

        # Return a valid JSON string with API_calls and an (ignored) bug_locations list.
        selected_apis = json.dumps({"API_calls": ["dummy_api_call"], "bug_locations": []})
        monkeypatch.setattr(agent_proxy, "run_with_retries", lambda resp: (selected_apis, []))

        # Monkey-patch parse_function_invocation to return a dummy function name and arguments.
        monkeypatch.setattr("app.search.search_manage.parse_function_invocation", lambda s: ("dummy_func", ["value1"]))

        # Define a dummy function with a self parameter.
        def dummy_func(self, arg1):
            return ("dummy result", None, True)
        # Attach dummy_func as a bound method to the backend.
        sm.backend.dummy_func = dummy_func.__get__(sm.backend, type(sm.backend))

        # Prevent actual printing.
        monkeypatch.setattr("app.search.search_manage.print_banner", lambda msg: None)
        monkeypatch.setattr("app.search.search_manage.print_acr", lambda msg, title: None)

        bug_locations, msg_thread = sm.search_iterative(dummy_task, sbfl_result, reproducer_result, None)
        # In this branch, since API calls were made, no bug locations are returned.
        assert bug_locations == []
        assert isinstance(msg_thread, DummyMessageThread)
        # Also check that the tool call layer has recorded the dummy API call.
        assert len(sm.tool_call_layers) == 1
        assert sm.tool_call_layers[0] == [
            {"func_name": "dummy_func", "arguments": {"arg1": "value1"}, "call_ok": True}
        ]