import json
import os
import shutil
from abc import ABC, abstractmethod
from os.path import join as pjoin
from pathlib import Path

import requests

from app import utils as app_utils
from app.log import log_and_print
from app.task import PlainTask, SweTask, Task

META_FILE_NAME = "meta.json"


class RawTask(ABC):
    @property
    @abstractmethod
    def task_id(self) -> str:
        raise NotImplementedError("abstract base class")

    @abstractmethod
    def to_task(self) -> Task:
        raise NotImplementedError("abstract base class")

    @abstractmethod
    def dump_meta_data(self, output_dir: str) -> None:
        raise NotImplementedError("abstract base class")


class RawSweTask(RawTask):
    """
    Encapsulate everything required to run one task.
    """

    def __init__(self, task_id: str, setup_info: dict, task_info: dict):
        # the instance id from SWE-bench
        self._task_id = task_id
        # setup_info (Dict): keys: ['repo_path', 'env_name', 'pre_install', 'install','test_cmd']
        self.setup_info = setup_info
        # task_info (Dict): keys: ['base_commit', 'hints_text', 'created_at',
        # 'test_patch', 'repo', 'problem_statement', 'version', 'instance_id',
        # 'FAIL_TO_PASS', 'PASS_TO_PASS', 'environment_setup_commit']
        self.task_info = task_info

    @property
    def task_id(self) -> str:
        return self._task_id

    def to_task(self) -> SweTask:
        task_id = self.task_id
        setup_info = self.setup_info
        task_info = self.task_info
        return SweTask(
            task_id=task_id,
            problem_statement=task_info["problem_statement"],
            repo_path=setup_info["repo_path"],
            env_name=setup_info["env_name"],
            pre_install_cmds=setup_info["pre_install"],
            install_cmd=setup_info["install"],
            # command to run the relevant tests,
            test_cmd=setup_info["test_cmd"],
            commit=task_info["base_commit"],
            repo_name=task_info["repo"],
            repo_version=task_info["version"],
            # modifications to the test suite for this task instance,
            test_patch=task_info["test_patch"],
            testcases_passing=task_info["PASS_TO_PASS"],
            testcases_failing=task_info["FAIL_TO_PASS"],
        )

    def dump_meta_data(self, output_dir: str):
        meta = {
            "task_id": self.task_id,
            "setup_info": self.setup_info,
            "task_info": self.task_info,
        }
        with open(pjoin(output_dir, META_FILE_NAME), "w") as f:
            json.dump(meta, f, indent=4)
        with open(pjoin(output_dir, "problem_statement.txt"), "w") as f:
            f.write(self.task_info["problem_statement"])
        with open(pjoin(output_dir, "developer_patch.diff"), "w") as f:
            f.write(self.task_info["patch"])


class RawGithubTask(RawTask):
    """
    Encapsulate everything required to run ACR on a fresh issue from the internet.
    """

    def __init__(
        self,
        task_id: str,
        clone_link: str,
        commit_hash: str | None,
        issue_link: str,
        setup_dir: str,
    ):
        self._task_id = task_id
        self.clone_link = clone_link
        self.commit_hash = commit_hash
        self.issue_link = issue_link
        self.setup_dir = setup_dir
        self.clone_path = pjoin(self.setup_dir, self.task_id)
        self.problem_statement, self.created_at = self.fetch_issue()
        self.clone_repo()

    @property
    def task_id(self) -> str:
        return self._task_id

    def clone_repo(self):
        clone_path = Path(self.clone_path)
        if os.path.exists(clone_path):
            log_and_print(
                f"Path {clone_path} already exists. Removing it to get a fresh clone."
            )
            shutil.rmtree(clone_path)
        app_utils.clone_repo(self.clone_link, str(clone_path.parent), clone_path.name)
        log_and_print(f"Cloned source code to {clone_path}.")
        if self.commit_hash is None:
            # let's get the current commit hash
            with app_utils.cd(clone_path):
                self.commit_hash = app_utils.get_current_commit_hash()

    def dump_meta_data(self, output_dir: str):
        meta = {
            "task_info": {
                "base_commit": self.commit_hash,
                "created_at": self.created_at,
                "problem_statement": self.problem_statement,
                "instance_id": self.task_id,
            },
            "setup_info": {
                "repo_path": self.clone_path,
            },
        }

        meta_file = pjoin(output_dir, META_FILE_NAME)

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=4)

    def fetch_issue(self):
        if "github.com" not in self.issue_link:
            raise NotImplementedError("Only GitHub issues are supported for now.")

        retrieved_issue = self.fetch_github_issue(self.issue_link)

        if retrieved_issue is None:
            raise RuntimeError(
                f"Failed to retrieve issue information from {self.issue_link}"
            )

        title, body, created_at = retrieved_issue

        problem_statement = f"{title}\n{body}"

        return problem_statement, created_at

    @classmethod
    def fetch_github_issue(cls, issue_url: str) -> tuple[str, str, str]:
        """Extract owner, repo, and issue number from the URL"""

        # Example issue URL: https://github.com/owner/repo/issues/123

        _, owner, repo, _, issue_number = issue_url.rsplit("/", 4)

        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        response = requests.get(api_url)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch issue information: {response.status_code}"
            )

        issue_info = response.json()

        title = issue_info["title"]
        body = issue_info["body"]
        created_at = issue_info["created_at"]

        return title, body, created_at

    def to_task(self) -> PlainTask:
        return PlainTask(
            commit_hash=self.commit_hash,
            local_path=self.clone_path,
            problem_statement=self.problem_statement,
        )


class RawLocalTask(RawTask):
    """
    Encapsulate everything required to run ACR on a local issue on the disk.
    """

    def __init__(self, task_id: str, local_repo: str, issue_file: str):
        self._task_id = task_id
        self.local_repo = local_repo
        self.issue_file = issue_file
        self.commit_hash = self.init_local_repo()
        self.problem_statement = self.read_issue_from_file()

    @property
    def task_id(self) -> str:
        return self._task_id

    def init_local_repo(self):
        with app_utils.cd(self.local_repo):
            if not app_utils.is_git_repo():
                # non git repo - let's make it a git repo first
                app_utils.initialize_git_repo_and_commit()
            commit = app_utils.get_current_commit_hash()
        return commit

    def read_issue_from_file(self) -> str:
        # ignore encoding errors so at least we can have some issue content
        issue = Path(self.issue_file).read_text(errors="ignore")
        return issue

    def dump_meta_data(self, output_dir: str):
        meta = {
            "task_info": {
                "base_commit": self.commit_hash,
                "problem_statement": self.problem_statement,
                "instance_id": self.task_id,
            },
            "setup_info": {
                "repo_path": self.local_repo,
            },
        }

        meta_file = pjoin(output_dir, META_FILE_NAME)

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=4)

    def to_task(self) -> PlainTask:
        return PlainTask(
            commit_hash=self.commit_hash,
            local_path=self.local_repo,
            problem_statement=self.problem_statement,
        )
