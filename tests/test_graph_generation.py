import pytest
from arigin.graph import elements

# Importing the functions to test from your module
from arigin.graph.generation import (
    extract_first_pattern, 
    extract_innermost_parentheses, 
    extract_multiplication_or_division, 
    extract_addition_subtraction,
    remove_redundant_parenthesis,
    graph_elements_from_primitive_expression
)


def test_extract_first_pattern():
    assert extract_first_pattern(r"\d+", "There are 123 apples") == "123"
    assert extract_first_pattern(r"[a-z]+", "Hello World") == "ello"
    assert extract_first_pattern(r"\s+", "   Hello") == "   "
    assert extract_first_pattern(r"\d+", "No numbers here") is None


def test_extract_innermost_parentheses():
    assert extract_innermost_parentheses("(a + (b - c))") == "(b - c)"
    assert extract_innermost_parentheses("((3 + 4) * (2 + 5))") == "(3 + 4)"
    assert extract_innermost_parentheses("No parentheses here") is None
    assert extract_innermost_parentheses("Single parentheses ()") == "()"


def test_extract_multiplication_or_division():
    assert extract_multiplication_or_division("5 * 10 + 2") == "5 * 10"
    assert extract_multiplication_or_division("a / b - c") == "a / b"
    assert extract_multiplication_or_division("2 + 3") is None


def test_extract_addition_subtraction():
    assert extract_addition_subtraction("5 + 10 * 2") == "5 + 10"
    assert extract_addition_subtraction("x - y / z") == "x - y"
    assert extract_addition_subtraction("2 * 3") is None


def test_remove_redundant_parenthesis():
    # Test cases with redundant parentheses around alphanumeric and integers
    assert remove_redundant_parenthesis("( ( R123 ) )") == "R123"
    assert remove_redundant_parenthesis("(( ( ( Ralpha ) ) ))") == "Ralpha"
    assert remove_redundant_parenthesis("( ( 42 ) )") == "42"

    # Test cases with no redundant parentheses
    assert remove_redundant_parenthesis("Rxyz + (2 * 3)") == "Rxyz + (2 * 3)"
    assert remove_redundant_parenthesis("10 + 20") == "10 + 20"
    assert remove_redundant_parenthesis("(R1 + R2)") == "(R1 + R2)"

    # Edge cases with complex nesting
    assert remove_redundant_parenthesis("((( ( 100 ) )))") == "100"
    assert remove_redundant_parenthesis("( ( ( Rabc ) ) )") == "Rabc"
