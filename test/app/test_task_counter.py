import pytest
import app.task_counter as tc


# Fixture to reset the counters before each test.
@pytest.fixture(autouse=True)
def reset_task_counter():
    tc.init_total_num_tasks(0)
    tc.num_completed_tasks.value = 0
    tc.init_total_num_task_groups(0)
    tc.num_completed_task_groups.value = 0


def test_init_total_num_tasks():
    """Verify that init_total_num_tasks correctly sets the global total_num_tasks."""
    tc.init_total_num_tasks(10)
    assert tc.total_num_tasks == 10


def test_incre_completed_tasks():
    """
    Verify that incre_completed_tasks correctly increments the global completed task counter.
    """
    tc.init_total_num_tasks(5)  # Set total tasks (for context)
    # Initially, the counter should be 0.
    assert tc.num_completed_tasks.value == 0
    count1 = tc.incre_completed_tasks()
    assert count1 == 1
    count2 = tc.incre_completed_tasks()
    assert count2 == 2


def test_incre_completed_task_groups():
    """
    Verify that incre_completed_task_groups correctly increments the global completed task groups counter.
    """
    tc.init_total_num_task_groups(3)
    # Initially, the counter should be 0.
    assert tc.num_completed_task_groups.value == 0
    count1 = tc.incre_completed_task_groups()
    assert count1 == 1
    count2 = tc.incre_completed_task_groups()
    assert count2 == 2


def test_incre_task_return_msg():
    """
    Verify that incre_task_return_msg increments the tasks counter and returns the correct formatted string.
    """
    tc.init_total_num_tasks(5)
    tc.init_total_num_task_groups(3)
    tc.num_completed_tasks.value = 0
    tc.num_completed_task_groups.value = 0

    msg1 = tc.incre_task_return_msg()
    expected1 = ">>> Completed 1/5 tasks. For groups, completed 0/3 so far."
    assert msg1 == expected1

    msg2 = tc.incre_task_return_msg()
    expected2 = ">>> Completed 2/5 tasks. For groups, completed 0/3 so far."
    assert msg2 == expected2


def test_incre_task_group_return_msg():
    """
    Verify that incre_task_group_return_msg increments the task groups counter and returns the correct message.
    """
    tc.init_total_num_task_groups(4)
    tc.num_completed_task_groups.value = 0

    msg1 = tc.incre_task_group_return_msg()
    expected1 = ">>>>>> Completed 1/4 task groups."
    assert msg1 == expected1

    msg2 = tc.incre_task_group_return_msg()
    expected2 = ">>>>>> Completed 2/4 task groups."
    assert msg2 == expected2
