import json
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from pprint import pformat

from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import (
    Function as OpenaiFunction,
)

from app import utils as apputils
from app.search import search_utils


@dataclass
class MethodId:
    class_name: str
    method_name: str

    def __str__(self):
        if self.class_name:
            return f"{self.class_name}.{self.method_name}"
        return self.method_name

    def __hash__(self):
        return hash((self.class_name, self.method_name))


class FunctionCallIntent:
    """An intent to call a tool function.

    This object created from OpenAI API response.
    """

    def __init__(
        self,
        func_name: str,
        arguments: Mapping[str, str],
        openai_func: OpenaiFunction | None,
    ):
        self.func_name = func_name
        self.arg_values = dict()
        self.arg_values.update(arguments)
        # record the original openai function object,
        # which is used when we want tell the model that it has
        # previously called this function/tool
        self.openai_func = openai_func or OpenaiFunction(
            arguments=json.dumps(arguments), name=func_name
        )

    def __str__(self):
        return f"FunctionCallIntent(func_name={self.func_name}, arguments={self.arg_values})"

    def to_dict(self):
        return {"func_name": self.func_name, "arguments": self.arg_values}

    def to_dict_with_result(self, call_ok: bool):
        return {
            "func_name": self.func_name,
            "arguments": self.arg_values,
            "call_ok": call_ok,
        }


class MessageThread:
    """
    Represents a thread of conversation with the model.
    Abstrated into a class so that we can dump this to a file at any point.
    """

    def __init__(self, messages=None):
        self.messages: list[dict] = messages or []

    def add(self, role: str, message: str):
        """
        Add a new message to the thread.
        Args:
            message (str): The content of the new message.
            role (str): The role of the new message.
        """
        self.messages.append({"role": role, "content": message})

    def add_system(self, message: str):
        self.messages.append({"role": "system", "content": message})

    def add_user(self, message: str):
        self.messages.append({"role": "user", "content": message})

    def add_tool(self, message: str, tool_call_id: str):
        m = {"role": "tool", "content": message, "tool_call_id": tool_call_id}
        self.messages.append(m)

    def add_model(
        self, message: str | None, tools: list[ChatCompletionMessageToolCall] = []
    ):
        # let's serialize tools into json first
        json_tools = []
        for tool in tools:
            this_tool_dict = {}
            this_tool_dict["id"] = tool.id
            this_tool_dict["type"] = tool.type
            # now serialize function as well
            func_obj: OpenaiFunction = tool.function
            func_args: str = func_obj.arguments
            func_name: str = func_obj.name
            this_tool_dict["function"] = {"name": func_name, "arguments": func_args}
            json_tools.append(this_tool_dict)

        if json_tools == []:
            # there is no tool calls from the model last time,
            # the best we could do is to return the generated text
            self.messages.append({"role": "assistant", "content": message})
        else:
            self.messages.append(
                {"role": "assistant", "content": None, "tool_calls": json_tools}
            )

    def to_msg(self) -> list[dict]:
        """
        Convert to the format to be consumed by the model.
        Returns:
            List[Dict]: The message thread.
        """
        return self.messages

    def __str__(self):
        return pformat(self.messages, width=160, sort_dicts=False)

    def save_to_file(self, file_path: str | PathLike):
        """
        Save the current state of the message thread to a file.
        Args:
            file_path (str): The path to the file.
        """
        Path(file_path).write_text(json.dumps(self.messages, indent=4))

    def get_round_number(self) -> int:
        """
        From the current message history, decide how many rounds have been completed.
        """
        completed_rounds = 0
        for message in self.messages:
            if message["role"] == "assistant":
                completed_rounds += 1
        return completed_rounds

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Load the message thread from a file.
        Args:
            file_path (str): The path to the file.
        Returns:
            MessageThread: The message thread.
        """
        with open(file_path) as f:
            messages = json.load(f)
        return cls(messages)


class ReproResult:
    # TODO: add exit code
    reproduced: bool
    stdout: str
    stderr: str
    returncode: int

    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.reproduced = returncode != 0 and "AssertionError" in stderr

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Reproduced: {self.reproduced}",
                "",
                "Stdout:",
                self.stdout,
                "",
                "Stderr:",
                self.stderr,
            ]
        )


@dataclass
class SearchResult:
    """Dataclass to hold search results."""

    # this is absolute path
    file_path: str
    # line numbers are 1-based
    start: int | None
    end: int | None
    class_name: str | None
    func_name: str | None
    code: str

    def to_tagged_upto_file(self, project_root: str):
        """Convert the search result to a tagged string, upto file path."""
        rel_path = apputils.to_relative_path(self.file_path, project_root)
        file_part = f"<file>{rel_path}</file>"
        return file_part

    def to_tagged_upto_class(self, project_root: str):
        """Convert the search result to a tagged string, upto class."""
        prefix = self.to_tagged_upto_file(project_root)
        class_part = (
            f"<class>{self.class_name}</class>" if self.class_name is not None else ""
        )
        return f"{prefix}\n{class_part}"

    def to_tagged_upto_func(self, project_root: str):
        """Convert the search result to a tagged string, upto function."""
        prefix = self.to_tagged_upto_class(project_root)
        func_part = (
            f" <func>{self.func_name}</func>" if self.func_name is not None else ""
        )
        return f"{prefix}{func_part}"

    def to_tagged_str(self, project_root: str):
        """Convert the search result to a tagged string."""
        prefix = self.to_tagged_upto_func(project_root)
        code_part = f"<code>\n{self.code}\n</code>"
        return f"{prefix}\n{code_part}"

    @staticmethod
    def collapse_to_file_level(lst, project_root: str) -> str:
        """Collapse search results to file level."""
        res = dict()  # file -> count
        for r in lst:
            if r.file_path not in res:
                res[r.file_path] = 1
            else:
                res[r.file_path] += 1
        res_str = ""
        for file_path, count in res.items():
            rel_path = apputils.to_relative_path(file_path, project_root)
            file_part = f"<file>{rel_path}</file>"
            res_str += f"- {file_part} ({count} matches)\n"
        return res_str

    @staticmethod
    def collapse_to_method_level(lst, project_root: str) -> str:
        """Collapse search results to method level."""
        res = dict()  # file -> dict(method -> count)
        for r in lst:
            if r.file_path not in res:
                res[r.file_path] = dict()
            func_str = r.func_name if r.func_name is not None else "Not in a function"
            if func_str not in res[r.file_path]:
                res[r.file_path][func_str] = 1
            else:
                res[r.file_path][func_str] += 1
        res_str = ""
        for file_path, funcs in res.items():
            rel_path = apputils.to_relative_path(file_path, project_root)
            file_part = f"<file>{rel_path}</file>"
            for func, count in funcs.items():
                if func == "Not in a function":
                    func_part = func
                else:
                    func_part = f" <func>{func}</func>"
                res_str += f"- {file_part}{func_part} ({count} matches)\n"
        return res_str


class BugLocation:
    rel_file_path: str
    abs_file_path: str
    # line numbers are 1-based
    start: int | None
    end: int | None

    class_name: str | None
    # NOTE: from patch generation onwards, call this method_name
    method_name: str | None

    code: str

    intended_behavior: str

    def __init__(
        self, search_res: SearchResult, project_path: str, intended_bebavior: str
    ):
        assert search_res.start is not None
        assert search_res.end is not None

        # turn a search result into bug location
        self.abs_file_path = search_res.file_path
        self.rel_file_path = apputils.to_relative_path(
            search_res.file_path, project_path
        )

        self.start = search_res.start
        self.end = search_res.end

        self.class_name = search_res.class_name
        self.method_name = search_res.func_name

        self.intended_behavior = intended_bebavior

        # we know the line numbers are reliable, so just get the actual
        # code here again to be safe
        self.code = search_utils.get_code_snippets(
            self.abs_file_path, self.start, self.end
        )

    def to_dict(self):
        return {
            "rel_file_path": self.rel_file_path,
            "abs_file_path": self.abs_file_path,
            "start": self.start,
            "end": self.end,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "code": self.code,
            "intended_behavior": self.intended_behavior,
        }

    def __eq__(self, other):
        return (
            self.rel_file_path == other.rel_file_path
            and self.start == other.start
            and self.end == other.end
        )

    def __hash__(self):
        return hash((self.rel_file_path, self.start, self.end))

    def __str__(self):
        return (
            f"<file>{self.rel_file_path}</file>\n"
            f"<class>{self.class_name}</class>\n"
            f"<method>{self.method_name}</method>\n"
            f"<code>\n{self.code}\n</code>"
            f"<intended_behavior>{self.intended_behavior}</intended_behavior>"
        )

    def __repr__(self):
        return self.__str__()

    def to_str_for_model(self):
        return self.__str__()

    @classmethod
    def multiple_locs_to_str_for_model(cls, locs: list["BugLocation"]):
        res = ""
        for idx, loc in enumerate(locs):
            actual_idx = idx + 1
            res += f"Location #{actual_idx}:\n"
            res += loc.to_str_for_model() + "\n\n"
        return res
