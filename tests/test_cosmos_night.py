"""The spoken list of objects: plain names, and nothing named twice."""
import cosmos


def test_a_bare_name_beside_a_fuller_one_is_dropped():
    assert cosmos.tidy(["water bottle", "blue cup", "orange", "laptop", "cup"]) == ["water bottle", "blue cup", "orange", "laptop"]
    assert cosmos.tidy(["laptop", "black laptop", "bottle", "water bottle"]) == ["black laptop", "water bottle"]


def test_names_are_made_plain_and_not_repeated():
    assert cosmos.tidy(["Laptop computer.", "laptop", "orange fruit", "Orange", "computer mouse"]) == ["laptop", "orange", "mouse"]


def test_different_things_with_a_shared_word_are_both_kept():
    assert cosmos.tidy(["blue cup", "green cup", "cupcake"]) == ["blue cup", "green cup", "cupcake"]
    assert cosmos.tidy(["a", "b", "c", "d", "e", "f"], n=5) == ["a", "b", "c", "d", "e"]
