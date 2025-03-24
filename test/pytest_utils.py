# test/pytest_util.py
from pathlib import Path

from app.data_structures import MessageThread
from app.task import Task
from openai import BadRequestError


# --- Dummy helper classes ---
class DummyMessageThread(MessageThread):
    def __init__(self):
        # minimal initialization
        pass

    def save_to_file(self, file_path):
        # simulate saving to file; write dummy content
        Path(file_path).write_text("dummy message thread content")


class DummyTask(Task):
    def __init__(self, project_path="dummy_project", issue="dummy issue"):
        # Allow project_path to be specified; default to "dummy_project"
        self._project_path = project_path

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


###############################################################################
# Dummy Model for Testing
###############################################################################
class DummyModel:
    def __init__(self, responses):
        self.responses = responses  # a list of response strings
        self.call_count = 0

    def setup(self):
        pass

    def call(self, messages, **kwargs):
        response = self.responses[self.call_count]
        self.call_count += 1
        return (response,)


# --- Section for common classes used when testing Models
# from test.pytest_utils import *


# Dummy classes to simulate the OpenAI response.
class DummyUsage:
    prompt_tokens = 1
    completion_tokens = 2


class DummyMessage:
    def __init__(self, content="Test response", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class DummyChoice:
    def __init__(self):
        self.message = DummyMessage()


class DummyResponse:
    def __init__(self):
        self.usage = DummyUsage()
        self.choices = [DummyChoice()]


class DummyCompletions:
    last_kwargs = {}  # initialize as a class attribute

    def create(self, *args, **kwargs):
        DummyCompletions.last_kwargs = kwargs  # capture the kwargs passed in
        return DummyResponse()


# Dummy client chat now includes a completions attribute.
class DummyClientChat:
    completions = DummyCompletions()


# Dummy client with a chat attribute.
class DummyClient:
    chat = DummyClientChat()


# Dummy thread cost container to capture cost updates.
class DummyThreadCost:
    process_cost = 0.0
    process_input_tokens = 0
    process_output_tokens = 0


# --- Dummy Response Object for BadRequestError ---
class DummyResponseObject:
    request = "dummy_request"
    status_code = 400  # Provide a dummy status code.
    headers = {"content-type": "application/json"}


class DummyThreadCost:
    process_cost = 0.0
    process_input_tokens = 0
    process_output_tokens = 0


# To test sys.exit in check_api_key failure.
class SysExitException(Exception):
    pass


# --- For testing BadRequestError handling ---
# expect RetryError as the last in the error chain, with BadRequestError before it


# Define a dummy error class to simulate BadRequestError with a code attribute.
class DummyBadRequestError(BadRequestError):
    def __init__(self, message):
        # Do not call super().__init__ to avoid unexpected keyword errors.
        self.message = message


# Parameterized dummy completions that always raises BadRequestError with the provided code.
class DummyBadRequestCompletions:
    def __init__(self, code: str):
        self.code = code

    def create(self, *args, **kwargs):
        print(f"DummyBadRequestCompletions.create called with code {self.code}")
        err = BadRequestError("error", response=DummyResponseObject(), body={})
        err.code = self.code
        raise err


# Dummy client chat that holds an instance of the dummy completions.
class DummyBadRequestClientChat:
    def __init__(self, code: str):
        self.completions = DummyBadRequestCompletions(code)


# Dummy client that uses the dummy client chat.
class DummyBadRequestClient:
    def __init__(self, code: str):
        self.chat = DummyBadRequestClientChat(code)


# Utility to extract the exception chain for inspection.
def extract_exception_chain(exc):
    """Utility to walk the __cause__ chain and return a list of exceptions."""
    chain = [exc]
    while exc.__cause__ is not None:
        exc = exc.__cause__
        chain.append(exc)
    return chain


# --- Section for dummy functions ---
def dummy_check_api_key(self):
    print("dummy_check_api_key called")
    return "dummy-key"


def dummy_sleep(seconds):
    print(f"dummy_sleep called with {seconds} seconds (disabled)")
    return None


def dummy_sys_exit(code):
    raise SysExitException(f"sys.exit called with {code}")
