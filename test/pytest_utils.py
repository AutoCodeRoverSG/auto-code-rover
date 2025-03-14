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
