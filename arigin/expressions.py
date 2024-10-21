import re
import random

OPERATORS = ["+", "-", "*", "/"]
OPEN_PARENTHESIS = "("
CLOSE_PARENTHESIS = ")"


def generate(max_integer: int, max_numbers: int) -> str:
    """
    Generate an arithmetic expression by applying the formatting rules,
    which are hardcoded to fill basic arithmetic rules.

    :param max_integer: Greatest integer to be used in the generated
                         result.
    :type max_integer: int
    :param max_numbers: maximum number of integers to be created. Must be
                       greater or equal 2.
    :type max_numbers: int

    :return: String representing the expression.
    """

    expression = []
    
    max_numbers = random.randint(2, max(max_numbers, 2))

    n_open_parentesis = 0
    n_numbers = 0

    def _number(max_integer: int):
        return random.randint(1, max_integer)

    while n_numbers < max_numbers:
        if len(expression):
            previous_entity = expression[-1]
        else:
            previous_entity = None

        if isinstance(previous_entity, int):  # Number
            next_possible_entity = list(OPERATORS)
            if n_open_parentesis > 0:
                next_possible_entity.append(CLOSE_PARENTHESIS)
            next_entity = random.choice(next_possible_entity)
        elif previous_entity in OPERATORS:  # Operator
            next_possible_entity = [_number(max_integer), OPEN_PARENTHESIS]
            next_entity = random.choice(next_possible_entity)
        elif previous_entity == CLOSE_PARENTHESIS:  # )
            next_possible_entity = list(OPERATORS)
            next_entity = random.choice(next_possible_entity)
        elif previous_entity == OPEN_PARENTHESIS:  # (
            next_possible_entity = [_number(max_integer), OPEN_PARENTHESIS]
            next_entity = random.choice(next_possible_entity)
        else:  # None <- initial
            next_possible_entity = [
                _number(max_integer),
                OPEN_PARENTHESIS
            ]
            next_entity = random.choice(next_possible_entity)

        if next_entity == CLOSE_PARENTHESIS:
            n_open_parentesis -= 1
        elif next_entity == OPEN_PARENTHESIS:
            n_open_parentesis += 1
        elif isinstance(next_entity, int):
            n_numbers += 1

        expression.append(next_entity)

    while n_open_parentesis > 0:
        expression += CLOSE_PARENTHESIS
        n_open_parentesis -= 1

    return " ".join((str(x) for x in expression))
