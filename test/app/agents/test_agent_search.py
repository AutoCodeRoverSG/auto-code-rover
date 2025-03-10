from unittest.mock import patch, MagicMock
import pytest
from collections.abc import Generator

from app.agents.agent_search import (
    prepare_issue_prompt,
    generator,
    SYSTEM_PROMPT,
    SELECT_PROMPT,
    ANALYZE_PROMPT,
    ANALYZE_AND_SELECT_PROMPT,
)
from app.data_structures import MessageThread

def test_prepare_issue_prompt():
    input_str = (
        "   This is a sample problem statement.   \n"
        "<!-- This is a comment that should be removed -->\n"
        "\n"
        "It spans multiple lines.\n"
        "   And has extra spaces.  \n"
        "\n"
        "<!-- Another comment\n"
        "still in comment -->\n"
        "Final line."
    )
    
    expected_output = (
        "<issue>This is a sample problem statement.\n"
        "It spans multiple lines.\n"
        "And has extra spaces.\n"
        "Final line.\n</issue>"
    )
    
    assert prepare_issue_prompt(input_str) == expected_output

@patch("app.agents.agent_search.common.SELECTED_MODEL", new_callable=MagicMock, create=True)
@patch("app.agents.agent_search.print_acr")
@patch("app.agents.agent_search.print_retrieval")
@patch("app.agents.agent_search.config")
def test_generator_retry(mock_config, mock_print_retrieval, mock_print_acr, mock_selected_model):
    """
    Test the generator branch where re_search is True.
    In this branch the generator will:
      1. Yield its first API selection response.
      2. Process a search result with re_search True (simulating a failed consumption),
         which adds the search result as a user message and restarts the loop.
      3. Yield a new API selection response.
    """
    # Set configuration flags.
    mock_config.enable_sbfl = False
    mock_config.reproduce_and_review = False

    # Provide two responses:
    #  - First API selection call.
    #  - Next iteration API selection call after the retry.
    mock_selected_model.call.side_effect = [
        ("API selection response",),
        ("API selection response after retry",)
    ]

    issue_stmt = "Sample issue"
    sbfl_result = ""
    reproducer_result = ""

    gen = generator(issue_stmt, sbfl_result, reproducer_result)

    res_text, _ = next(gen)
    assert res_text == "API selection response"

    search_result = "Retry search result"
    res_text_retry, msg_thread_retry = gen.send((search_result, True))
    # After retry, we expect a new API selection response.
    assert res_text_retry == "API selection response after retry"
    # Verify that the search result was added to the message thread as a user message.
    user_msgs = [m for m in msg_thread_retry.messages if m.get("role") == "user"]
    assert any(search_result in m.get("content", "") for m in user_msgs)
