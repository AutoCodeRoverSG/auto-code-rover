import json
import os
import time
from os.path import join as pjoin
from pathlib import Path

from app import utils as app_utils
from app.log import log_and_print
from app.raw_tasks import PlainTask, RawTask


class RawGithubTask_for_debug(RawTask):
    """
    Encapsulate everything required to run ACR on a fresh issue from the internet.

    For debug / dev: github limits the calling times of api. Therefore when
    debuging, use the hardcode task:
        self.clone_link = 'https://github.com/langchain-ai/langchain.git'
        self.commit_hash = 'cb6e5e5'
        self.issue_link = 'https://github.com/langchain-ai/langchain/issues/20453'
        self.problem_statement: refer to the implementation of fetch_github_issue function
    """

    def __init__(
        self,
        task_id: str,
        clone_link: str,
        commit_hash: str,
        issue_link: str,
        setup_dir: str,
    ):
        self._task_id = task_id
        self.clone_link = "https://github.com/langchain-ai/langchain.git"
        self.commit_hash = "cb6e5e5"
        self.issue_link = "https://github.com/langchain-ai/langchain/issues/20453"
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
            return  # Dev/Debug mode
        app_utils.clone_repo(self.clone_link, str(clone_path.parent), clone_path.name)
        log_and_print(f"Cloned source code to {clone_path}.")

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

        meta_file = pjoin(output_dir, "meta.json")

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

        title = """`PaiEasChatEndpoint._call_eas` should return `bytes` type instead of `str` type"""
        body = """
### Checked other resources

- [X] I added a very descriptive title to this issue.
- [X] I searched the LangChain documentation with the integrated search.
- [X] I used the GitHub search to find a similar question and didn't find it.
- [X] I am sure that this is a bug in LangChain rather than my code.
- [X] The bug is not resolved by updating to the latest stable version of LangChain (or the specific integration package).

### Example Code

```python
from langchain_community.chat_models import PaiEasChatEndpoint
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
prompt_value = prompt.invoke("test_message")

eas_chat_endpoint = PaiEasChatEndpoint(
    eas_service_url="your_service_url",
    eas_service_token="your_service_token"
)
eas_chat_endpoint._call(prompt_value.messages)
```

It is just example code that has a potential error, so I'll explain why it's a possible type error in the description.

### Error Message and Stack Trace (if applicable)

_No response_

### Description

https://github.com/langchain-ai/langchain/blob/b66a4f48fa5656871c3e849f7e1790dfb5a4c56b/libs/community/langchain_community/chat_models/pai_eas_endpoint.py#L193-L230

In the method `PaiEasChatEndpoint._call` at line 203, `self._call_eas(request_payload)` returns `str` type because it returns response.text at line 230 (https://requests.readthedocs.io/en/latest/api/#requests.Response.text).
Then, first argument of `_format_response_payload` can be `str` type at line 204, resulting in type mismatch.

https://github.com/langchain-ai/langchain/blob/b66a4f48fa5656871c3e849f7e1790dfb5a4c56b/libs/community/langchain_community/chat_models/pai_eas_endpoint.py#L167-L179

Moreover, if `JSONDecodeError` occurs in this method, then attribute error occurs at line 178 (`AttributeError: 'str' object has no attribute 'decode'`) because `output` variable is str type.
I think `PaiEasChatEndpoint._call_eas` should be fixed to return bytes type.
If I'm mistaken, I'd appreciate it if you could let me know.

### System Info

System Information
------------------
> OS:  Linux
> OS Version:  #190-Ubuntu SMP Fri Feb 2 23:24:22 UTC 2024
> Python Version:  3.9.18 (main, Feb 13 2024, 14:37:08)
[GCC 9.4.0]

Package Information
-------------------
> langchain_core: 0.1.42
> langchain: 0.1.16
> langchain_community: 0.0.32
> langsmith: 0.1.47
> langchain_text_splitters: 0.0.1

Packages not installed (Not Necessarily a Problem)
--------------------------------------------------
The following packages were not found:

> langgraph
> langserve
"""
        created_at = """2024-04-15T07:47:06Z"""

        return title, body, created_at

    def to_task(self) -> PlainTask:
        return PlainTask(
            commit_hash=self.commit_hash,
            local_path=self.clone_path,
            problem_statement=self.problem_statement,
        )


test_data = [
    {
        "title": "AutoCodeRover (context retrieval round 0)",
        "category": "auto_code_rover",
        "message": """Based on the files, classes, methods, and code statements from the issue related to the bug, you can use the following search APIs to get more context of the project.
- search_class(class_name: str): Search for a class in the codebase
- search_method_in_file(method_name: str, file_path: str): Search for a method in a given file
- search_method_in_class(method_name: str, class_name: str): Search for a method in a given class
- search_method(method_name: str): Search for a method in the entire codebase
- search_code(code_str: str): Search for a code snippet in the entire codebase
- search_code_in_file(code_str: str, file_path: str): Search for a code snippet in a given file file

Note that you can use multiple search APIs in one round.

Now analyze the issue and select necessary APIs to get more context of the project. Each API call must have concrete arguments as inputs.""",
    },
    {
        "title": "Context Retrieval Agent (round 0)",
        "category": "context_retrieval_agent",
        "message": """I will start by searching for the `PaiEasChatEndpoint` class to understand its structure and methods. Then, I will search for the `_call_eas` method within this class to investigate its implementation.

1. `search_class("PaiEasChatEndpoint")`
2. `search_method_in_class("_call_eas", "PaiEasChatEndpoint")`""",
    },
    {
        "title": "AutoCodeRover (Agent-selected API calls)",
        "category": "auto_code_rover",
        "message": """
            API calls:

- `search_class('PaiEasChatEndpoint')`

- `search_method_in_class('_call_eas', 'PaiEasChatEndpoint')`
            """,
    },
    {
        "title": "AutoCodeRover (context retrieval round 0)",
        "category": "auto_code_rover",
        "message": '''
            Result of search_class('PaiEasChatEndpoint'):

Found 1 classes with name PaiEasChatEndpoint in the codebase:

- Search result 1:
```
<file>libs/community/langchain_community/chat_models/pai_eas_endpoint.py</file>
<class>PaiEasChatEndpoint</class>
<code>
class PaiEasChatEndpoint(BaseChatModel):
@root_validator()
def validate_environment(cls, values: Dict) -> Dict:
@property
def _identifying_params(self) -> Dict[str, Any]:
@property
def _llm_type(self) -> str:
@property
def _default_params(self) -> Dict[str, Any]:
def _invocation_params(
self, stop_sequences: Optional[List[str]], **kwargs: Any
) -> dict:
def format_request_payload(
self, messages: List[BaseMessage], **model_kwargs: Any
) -> dict:
def _format_response_payload(
self, output: bytes, stop_sequences: Optional[List[str]]
) -> str:
def _generate(
self,
messages: List[BaseMessage],
stop: Optional[List[str]] = None,
run_manager: Optional[CallbackManagerForLLMRun] = None,
**kwargs: Any,
) -> ChatResult:
def _call(
self,
messages: List[BaseMessage],
stop: Optional[List[str]] = None,
run_manager: Optional[CallbackManagerForLLMRun] = None,
**kwargs: Any,
) -> str:
def _call_eas(self, query_body: dict) -> Any:
def _call_eas_stream(self, query_body: dict) -> Any:
def _convert_chunk_to_message_message(
self,
chunk: str,
) -> AIMessageChunk:

</code>
```


Result of search_method_in_class('_call_eas', 'PaiEasChatEndpoint'):

Found 1 methods with name _call_eas in class PaiEasChatEndpoint:

- Search result 1:
```
<file>libs/community/langchain_community/chat_models/pai_eas_endpoint.py</file>
<class>PaiEasChatEndpoint</class> <func>_call_eas</func>
<code>
def _call_eas(self, query_body: dict) -> Any:
"""Generate text from the eas service."""
headers = {
"Content-Type": "application/json",
"Accept": "application/json",
"Authorization": f"{self.eas_service_token}",
}

# make request
response = requests.post(
self.eas_service_url, headers=headers, json=query_body, timeout=self.timeout
)

if response.status_code != 200:
raise Exception(
f"Request failed with status code {response.status_code}"
f" and message {response.text}"
)

return response.text

</code>
```
            ''',
    },
    {
        "title": "Context Retrieval Agent (round 0)",
        "category": "auto_code_rover",
        "message": """Let's analyze collected context first""",
    },
    {
        "title": "AutoCodeRover (context retrieval round 0)",
        "category": "context_retrieval_agent",
        "message": """Based on the search results, here is the analysis of the `PaiEasChatEndpoint` class and the `_call_eas` method:

1. **PaiEasChatEndpoint Class**:
- The `PaiEasChatEndpoint` class is defined in the file `libs/community/langchain_community/chat_models/pai_eas_endpoint.py`.
- It inherits from `BaseChatModel`.
- The class contains various methods such as `_invocation_params`, `format_request_payload`, `_format_response_payload`, `_generate`, `_call`, `_call_eas`, `_call_eas_stream`, and `_convert_chunk_to_message_message`.
- The `_call` method is the one that calls the `_call_eas` method and currently returns a `str` type.

2. **_call_eas Method**:
- The `_call_eas` method is responsible for generating text from the EAS service.
- It takes a `query_body` parameter of type `dict`.
- It makes a POST request to the EAS service using the provided URL, headers, and query body.
- If the request fails (status code other than 200), it raises an exception with the status code and response message.
- Currently, the method returns the response text as a `str`.

Based on this analysis, the issue is indeed related to the `_call_eas` method returning a `str` type instead of `bytes`, which leads to a type mismatch in the `_call` method. The fix involves modifying the `_call_eas` method to return `bytes` instead of `str`.""",
    },
    {
        "title": "AutoCodeRover (context retrieval round 0)",
        "category": "auto_code_rover",
        "message": """Based on your analysis, answer below questions:
- do we need more context: construct search API calls to get more context of the project. (leave it empty if you don't need more context)
- where are bug locations: buggy files and methods. (leave it empty if you don't have enough information)""",
    },
    {
        "title": "AutoCodeRover (context retrieval round 0)",
        "category": "auto_code_rover",
        "message": """
        Based on the files, classes, methods, and code statements from the issue related to the bug, you can use the following search APIs to get more context of the project.
- search_class(class_name: str): Search for a class in the codebase
- search_method_in_file(method_name: str, file_path: str): Search for a method in a given file
- search_method_in_class(method_name: str, class_name: str): Search for a method in a given class
- search_method(method_name: str): Search for a method in the entire codebase
- search_code(code_str: str): Search for a code snippet in the entire codebase
- search_code_in_file(code_str: str, file_path: str): Search for a code snippet in a given file file

Note that you can use multiple search APIs in one round.

Now analyze the issue and select necessary APIs to get more context of the project. Each API call must have concrete arguments as inputs.
            """,
    },
    {
        "title": "Context Retrieval Agent (round 1)",
        "category": "context_retrieval_agent",
        "message": """- **More Context**:
- We have sufficient context to proceed with fixing the bug. No additional search API calls are needed at this point.

- **Bug Locations**:
- Buggy File: `libs/community/langchain_community/chat_models/pai_eas_endpoint.py`
- Buggy Method: `_call_eas` in the `PaiEasChatEndpoint` class""",
    },
    {
        "title": "AutoCodeRover (Agent-selected API calls)",
        "category": "auto_code_rover",
        "message": """

Bug locations

- file: `libs/community/langchain_community/chat_models/pai_eas_endpoint.py`, class: `PaiEasChatEndpoint`, method: `_call_eas`""",
    },
    {
        "title": "AutoCodeRover (Patch generation round 1)",
        "category": "auto_code_rover",
        "message": '''Here is the code in buggy locations:



Found 1 methods with name _call_eas in class PaiEasChatEndpoint:

- Search result 1:
```
<file>libs/community/langchain_community/chat_models/pai_eas_endpoint.py</file>
<class>PaiEasChatEndpoint</class> <func>_call_eas</func>
<code>
def _call_eas(self, query_body: dict) -> Any:
"""Generate text from the eas service."""
headers = {
"Content-Type": "application/json",
"Accept": "application/json",
"Authorization": f"{self.eas_service_token}",
}

# make request
response = requests.post(
self.eas_service_url, headers=headers, json=query_body, timeout=self.timeout
)

if response.status_code != 200:
raise Exception(
f"Request failed with status code {response.status_code}"
f" and message {response.text}"
)

return response.text

</code>
```
''',
    },
    {
        "title": "AutoCodeRover (patch generation)",
        "category": "auto_code_rover",
        "message": """
            Write a patch for the issue, based on the retrieved context.

You can import necessary libraries.


Return the patch in the format below.

Within `<file></file>`, replace `...` with actual file path.

Within `<original></original>`, replace `...` with the original code snippet from the program.

Within `<patched></patched>`, replace `...` with the fixed version of the original code. When adding orignal code and updated code, pay attention to indentation, as the code is in Python.
You can write multiple modifications if needed.

```
# modification 1
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 2
<file>...</file>
<original>...</original>
<patched>...</patched>

# modification 3
...
```""",
    },
    {
        "category": "patch_generation",
        "title": "Patch Generation (try 1 / 3)",
        "message": '''```
# Patch for changing the return type of _call_eas method to bytes

import requests

# modification 1
<file>libs/community/langchain_community/chat_models/pai_eas_endpoint.py</file>
<original>
def _call_eas(self, query_body: dict) -> Any:
"""Generate text from the eas service."""
headers = {
"Content-Type": "application/json",
"Accept": "application/json",
"Authorization": f"{self.eas_service_token}",
}

# make request
response = requests.post(
self.eas_service_url, headers=headers, json=query_body, timeout=self.timeout
)

if response.status_code != 200:
raise Exception(
f"Request failed with status code {response.status_code}"
f" and message {response.text}"
)

return response.text
</original>
<patched>
def _call_eas(self, query_body: dict) -> bytes:
"""Generate text from the eas service."""
headers = {
"Content-Type": "application/json",
"Accept": "application/json",
"Authorization": f"{self.eas_service_token}",
}

# make request
response = requests.post(
self.eas_service_url, headers=headers, json=query_body, timeout=self.timeout
)

if response.status_code != 200:
raise Exception(
f"Request failed with status code {response.status_code}"
f" and message {response.text}"
)

return response.content
</patched>
```''',
    },
    {
        "title": "AutoCodeRover (extracted patch)",
        "category": "auto_code_rover",
        "message": '''```diff
diff --git a/libs/community/langchain_community/chat_models/pai_eas_endpoint.py b/libs/community/langchain_community/chat_models/pai_eas_endpoint.py
index e438ad25e..309c6d287 100644
--- a/libs/community/langchain_community/chat_models/pai_eas_endpoint.py
+++ b/libs/community/langchain_community/chat_models/pai_eas_endpoint.py
@@ -208,7 +208,7 @@ class PaiEasChatEndpoint(BaseChatModel):

return generated_text

- def _call_eas(self, query_body: dict) -> Any:
+ def _call_eas(self, query_body: dict) -> bytes:
"""Generate text from the eas service."""
headers = {
"Content-Type": "application/json",
@@ -227,7 +227,7 @@ class PaiEasChatEndpoint(BaseChatModel):
f" and message {response.text}"
)

- return response.text
+ return response.content

def _call_eas_stream(self, query_body: dict) -> Any:
"""Generate text from the eas service."""

```''',
    },
    {
        "title": "AutoCodeRover (patch generation try 1 / 3)",
        "category": "auto_code_rover",
        "message": """Extracted a patch. Since validation is disabled, you should validation the patch later on. Ending the workflow.""",
    },
]


def test_generate_data(callback):
    for data in test_data:
        callback(data)
        time.sleep(1)
