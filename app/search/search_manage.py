import inspect
import json
from collections.abc import Mapping
from os.path import join as pjoin
from pathlib import Path

from loguru import logger

from app import config
from app.agents import agent_proxy, agent_search
from app.data_structures import BugLocation, MessageThread
from app.log import print_acr, print_banner
from app.search.search_backend import SearchBackend
from app.task import Task
from app.utils import parse_function_invocation


class SearchManager:
    def __init__(self, project_path: str, output_dir: str):
        # output dir for writing search-related things
        self.output_dir = pjoin(output_dir, "search")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # record the search APIs being used, in each layer
        self.tool_call_layers: list[list[Mapping]] = []

        self.backend: SearchBackend = SearchBackend(project_path)

    def search_iterative(
        self,
        task: Task,
        sbfl_result: str,
        reproducer_result: str,
        reproduced_test_content: str | None,
    ) -> tuple[list[BugLocation], MessageThread]:
        """
        Main entry point of the search manager.
        Returns:
            - Bug location info, which is a list of (code, intended behavior)
            - Class context code as string, or None if there is no context
            - The message thread that contains the search conversation.
        """
        search_api_generator = agent_search.generator(
            task.get_issue_statement(), sbfl_result, reproducer_result
        )
        # input to generator, should be (search_result_msg, re_search)
        # the first item is the results of search sent from backend
        # the second item is whether the agent should select APIs again, or proceed to analysis
        generator_input = None

        round_no = 0

        search_msg_thread: MessageThread | None = None  # for typing

        # TODO: change the global number to be local, since it's only for search
        for round_no in range(config.conv_round_limit):
            self.start_new_tool_call_layer()

            print_banner(f"CONTEXT RETRIEVAL ROUND {round_no}")

            # invoke agent search to choose search APIs
            agent_search_response, search_msg_thread = search_api_generator.send(
                generator_input
            )
            # print_retrieval(agent_search_response, f"round {round_no}")

            conversation_file = Path(self.output_dir, f"search_round_{round_no}.json")
            # save current state before starting a new round
            search_msg_thread.save_to_file(conversation_file)

            # extract json API calls from the raw response.
            selected_apis, proxy_threads = agent_proxy.run_with_retries(
                agent_search_response
            )

            logger.debug("Agent proxy return the following json: {}", selected_apis)

            proxy_msg_log = Path(self.output_dir, f"agent_proxy_{round_no}.json")
            proxy_messages = [thread.to_msg() for thread in proxy_threads]
            proxy_msg_log.write_text(json.dumps(proxy_messages, indent=4))

            if selected_apis is None:
                # agent search response could not be propagated to backend;
                # ask it to retry
                logger.debug(
                    "Could not extract API calls from agent search response, asking search agent to re-generate response."
                )
                search_result_msg = "The search API calls seem not valid. Please check the arguments you give carefully and try again."
                generator_input = (search_result_msg, True)
                continue

            # there are valid search APIs - parse them
            selected_apis_json: dict = json.loads(selected_apis)

            json_api_calls = selected_apis_json.get("API_calls", [])
            buggy_locations = selected_apis_json.get("bug_locations", [])

            self._print_selected_apis(json_api_calls, buggy_locations)

            # locations are confirmed by the agent - let's see whether the bug
            # locations are valid/precise
            if buggy_locations and (not json_api_calls):
                new_bug_locations = self._process_bug_locations(buggy_locations)

                if new_bug_locations:
                    return new_bug_locations, search_msg_thread

                # bug location is not precise enough to go into patch gen
                # let's prepare some message to be send to agent search
                # and go into next round
                logger.debug(
                    "Failed to retrieve code from all bug locations. Asking search agent to re-generate response."
                )
                search_result_msg = "Failed to retrieve code from all bug locations. You may need to check whether the arguments are correct or issue more search API calls."
                generator_input = (search_result_msg, True)
                continue

            # location not confirmed by the search agent - send backend result and go to next round
            collated_search_res_str = self._invoke_api_calls(json_api_calls)

            print_acr(collated_search_res_str, f"context retrieval round {round_no}")
            # send the results back to the search agent
            logger.debug(
                "Obtained search results from API invocation. Going into next retrieval round."
            )
            search_result_msg = collated_search_res_str
            generator_input = (search_result_msg, False)

        # used up all the rounds, but could not return the buggy locations
        logger.info("Too many rounds. Try writing patch anyway.")
        assert search_msg_thread is not None
        return [], search_msg_thread

    @staticmethod
    def _print_selected_apis(json_api_calls: list, buggy_locations: list) -> None:
        formatted = []
        if json_api_calls:
            formatted.append("API calls:")
            for call in json_api_calls:
                formatted.extend([f"\n- `{call}`"])

        if buggy_locations:
            formatted.append("\n\nBug locations")
            for location in buggy_locations:
                s = ", ".join(f"{k}: `{v}`" for k, v in location.items())
                formatted.extend([f"\n- {s}"])

        print_acr("\n".join(formatted), "Agent-selected API calls")

    def _process_bug_locations(self, buggy_locations: list) -> list[BugLocation]:
        """
        Transform agent-selected bug locations into BugLocation objects.
        Returns an empty list if no location could be extracted.
        """
        # dump the locations for debugging
        bug_loc_file = Path(self.output_dir, "bug_locations_before_process.json")
        bug_loc_file.write_text(json.dumps(buggy_locations, indent=4))

        new_bug_locations: list[BugLocation] = list()

        for loc in buggy_locations:
            # this is the transformed bug location
            new_bug_locations.extend(self.backend.get_bug_loc_snippets_new(loc))

        # remove duplicates in the bug locations
        unique_bug_locations: list[BugLocation] = []
        for loc in new_bug_locations:
            if loc not in unique_bug_locations:
                unique_bug_locations.append(loc)

        if not new_bug_locations:
            return new_bug_locations

        # some locations can be extracted, good to proceed to patch gen
        bug_loc_file_processed = Path(
            self.output_dir, "bug_locations_after_process.json"
        )

        json_obj = [loc.to_dict() for loc in new_bug_locations]
        bug_loc_file_processed.write_text(json.dumps(json_obj, indent=4))

        logger.debug(f"Bug location extracted successfully: {new_bug_locations}")

        return new_bug_locations

    def _invoke_api_calls(self, json_api_calls: list) -> str:
        """
        Invoke the search API calls on the backend and collate their results.
        """
        collated_search_res_str = ""

        for api_call in json_api_calls:
            func_name, func_args = parse_function_invocation(api_call)
            # TODO: there are currently duplicated code here and in agent_proxy.
            func_unwrapped = getattr(self.backend, func_name)
            while "__wrapped__" in func_unwrapped.__dict__:
                func_unwrapped = func_unwrapped.__wrapped__
            arg_spec = inspect.getfullargspec(func_unwrapped)
            arg_names = arg_spec.args[1:]  # first parameter is self

            assert len(func_args) == len(
                arg_names
            ), f"Number of argument is wrong in API call: {api_call}"

            kwargs = dict(zip(arg_names, func_args))

            function = getattr(self.backend, func_name)
            result_str, _, call_ok = function(**kwargs)
            collated_search_res_str += f"Result of {api_call}:\n\n"
            collated_search_res_str += result_str + "\n\n"

            # record the api calls made and the call status
            self.add_tool_call_to_curr_layer(func_name, kwargs, call_ok)

        return collated_search_res_str

    def start_new_tool_call_layer(self):
        self.tool_call_layers.append([])

    def add_tool_call_to_curr_layer(
        self, func_name: str, args: dict[str, str], result: bool
    ):
        self.tool_call_layers[-1].append(
            {
                "func_name": func_name,
                "arguments": args,
                "call_ok": result,
            }
        )

    def dump_tool_call_layers_to_file(self):
        """Dump the layers of tool calls to a file."""
        tool_call_file = Path(self.output_dir, "tool_call_layers.json")
        tool_call_file.write_text(json.dumps(self.tool_call_layers, indent=4))


# if __name__ == "__main__":
#     manager = SearchManager("/tmp", "/tmp/one")
#     func_name = "search_code"
#     func_args = {"code_str": "_separable"}

#     # func_name = "search_class"
#     # func_args = {"class_name": "ABC"}

#     function = getattr(manager.backend, func_name)

#     while "__wrapped__" in function.__dict__:
#         function = function.__wrapped__
#     arg_spec = inspect.getfullargspec(function)

#     print(arg_spec)
#     arg_names = arg_spec.args[1:]  # first parameter is self
#     kwargs = func_args

#     orig_func = getattr(manager.backend, func_name)
#     search_result, _, call_ok = orig_func(**kwargs)
