# content of test_sample.py
def func(x):
    return x + 1


def test_answer_unequal():
    assert func(3) != 5

def test_answer_equal():
    assert func(4) == 5