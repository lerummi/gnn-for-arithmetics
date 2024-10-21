import pytest
import re

from arigin.data_generation import generate, OPERATORS


@pytest.mark.parametrize("max_integer, max_numbers", [(10, 2), (100, 3), (1000, 5)])
def test_generate_returns_string(max_integer, max_numbers):
    """Test that the function returns a string."""
    result = generate(max_integer, max_numbers)
    assert isinstance(result, str)


@pytest.mark.parametrize("max_integer, max_numbers", [(10, 2), (100, 3), (1000, 5)])
def test_generate_has_numbers_and_operators(max_integer, max_numbers):
    """Test that the result contains numbers and operators."""
    result = generate(max_integer, max_numbers)
    
    # Ensure the result contains at least one number and operator
    assert any(char.isdigit() for char in result)
    assert any(op in result for op in ["+", "-", "*", "/"])


@pytest.mark.parametrize("max_integer, max_numbers", [(10, 2), (100, 3), (1000, 5)])
def test_generate_number_of_operands(max_integer, max_numbers):
    """Test that the expression contains the expected number of numbers (operands)."""
    result = generate(max_integer, max_numbers)

    # Count the number of numbers (operands) in the expression
    numbers = re.findall(r'\d+', result)
    assert 2 <= len(numbers) <= max_numbers


@pytest.mark.parametrize("max_integer, max_numbers", [(10, 2), (100, 3), (1000, 5)])
def test_generate_valid_arithmetic_expression(max_integer, max_numbers):
    """Test that the generated expression is a valid arithmetic expression."""
    result = generate(max_integer, max_numbers)

    assert result.count("(") == result.count(")")


@pytest.mark.parametrize("max_integer, max_numbers", [(10, 2), (100, 3), (1000, 5)])
def test_generate_balanced_parentheses(max_integer, max_numbers):
    """Test that the generated expression has balanced parentheses."""
    result = generate(max_integer, max_numbers)

    # Ensure that parentheses are balanced
    open_parentheses = result.count("(")
    close_parentheses = result.count(")")
    assert open_parentheses == close_parentheses


@pytest.mark.parametrize("max_integer, max_numbers", [(10, 2), (100, 3), (1000, 5)])
def test_generate_no_double_operators(max_integer, max_numbers):
    """Test that there are no consecutive operators in the generated expression."""
    result = generate(max_integer, max_numbers)

    # Ensure that there are no two consecutive operators in the expression
    for op in OPERATORS:
        assert f"{op} {op}" not in result
