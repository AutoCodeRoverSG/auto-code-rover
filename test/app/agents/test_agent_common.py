from app.data_structures import MessageThread
from app.agents.agent_common import replace_system_prompt, InvalidLLMResponse


def test_replace_system_prompt():
    # Setup: create a MessageThread with a system message and another message
    original_prompt = "Original System Prompt"
    new_prompt = "New System Prompt"
    messages = [
        {"role": "system", "content": original_prompt},
        {"role": "user", "content": "Hello"},
    ]
    msg_thread = MessageThread(messages=messages)

    # Execute: replace the system prompt
    updated_thread = replace_system_prompt(msg_thread, new_prompt)

    # Verify: first message should now have the new prompt
    assert (
        updated_thread.messages[0]["content"] == new_prompt
    ), "System prompt was not replaced correctly."
    # Verify: the rest of the messages remain unchanged
    assert (
        updated_thread.messages[1]["content"] == "Hello"
    ), "User message was unexpectedly modified."


def test_replace_system_prompt_returns_same_object():
    # Setup: create a MessageThread with a single system message
    messages = [{"role": "system", "content": "Initial Prompt"}]
    msg_thread = MessageThread(messages=messages)
    new_prompt = "Updated Prompt"

    # Execute: update the system prompt
    result = replace_system_prompt(msg_thread, new_prompt)

    # Verify: the same MessageThread instance is returned (in-place modification)
    assert (
        result is msg_thread
    ), "replace_system_prompt should return the same MessageThread object."
