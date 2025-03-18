# test/pytest_util.py
from pathlib import Path

from app.data_structures import MessageThread
from app.task import Task

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

# --- Section for dummy functions ---
def dummy_check_api_key(self):
    print("dummy_check_api_key called")
    return "dummy-key"

def dummy_sleep(seconds):
    print(f"dummy_sleep called with {seconds} seconds (disabled)")
    return None

def dummy_sys_exit(code):
    raise SysExitException(f"sys.exit called with {code}")
