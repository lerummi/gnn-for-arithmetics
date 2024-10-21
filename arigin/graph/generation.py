import re
from typing import Dict, List, Union, Optional
from arigin.graph import elements


# Placeholder for storing graph entities, i.e. dictionary of nodes
# and relationships
GraphEntities = Dict[
    str, 
    List[Union[elements.Node, elements.Relationship]]
    ]


def extract_first_pattern(pattern: str, expression: str) -> Optional[str]:
    """
    Given a string expression extract the first occurence of
    pattern within the string. Returns None if pattern is not found.
    """

    match = re.search(pattern, expression)

    if match:
        return match[0]

def extract_innermost_parentheses(expression: str) -> Optional[str]:
    """
    Given an expression as string, return the first and innermost
    expression between parenthesis '(' and ')'.
    """
    return extract_first_pattern(r"\([^()]*\)", expression)


def extract_multiplication_or_division(expression: str) -> Optional[str]:
    """
    Given a expression, extract the first part connected to
    muliplication or division. Returns primitive expression of the 
    form
        '[Number] [Operator] [Number]' 
    """
    return extract_first_pattern(r"(\w+\s*[*/]\s*\w+)", expression)


def extract_addition_subtraction(expression: str) -> Optional[str]:
    """
    Given a expression, extract the first part connected to
    addition and subtraction. Returns primitive expression of the 
    form
        '[Number] [Operator] [Number]' 
    """
    return extract_first_pattern(r"(\w+\s*[+-]\s*\w+)", expression)


def remove_redundant_parenthesis(expression: str) -> str:
    """
    Replace patterns like '( R<alphanumeric> )' -> 'R<alphanumeric>' and
    '( <integer> )' -> '<integer>'  and so remove redundant parentheses. 
    This also handles cases where there are multiple layers of redundant
    parentheses around R<alphanumeric>.

    :param expression: The mathematical expression as a string.
    :type expression: str
    :return: The cleaned expression with redundant parentheses removed.
    :rtype: str
    """

    # Regular expression to match redundant parentheses around R<alphanumeric>
    patterns = [
        r"\(\s*(R\w+)\s*\)",  #  ( R<alphanumeric> )
        r"\(\s*(\d+)\s*\)"  # ( <integer> )
    ]

    # Continuously remove redundant parentheses as long as they exist
    for pattern in patterns:
        while re.search(pattern, expression):
            expression = re.sub(pattern, r'\1', expression)

    return expression


def graph_elements_from_primitive_expression(
        primitive_expr: str,
        graph_entities: GraphEntities) -> elements.Result:
    """
    Create graph elements from a given primitive expression and update the 
    graph entities.

    A primitive expression is expected to be in the form:
    '[Number] [Operator] [Number]' or
    'R<alphanumeric> [Operator] R<alphanumeric>' or a mixture of both, 
    where 'R' represents a relationship to a node in the graph.

    This function will:
    - Parse the primitive expression into its components (left operand, 
      operator, right operand).
    - Create corresponding node elements for numbers or retrieve existing 
      nodes for relationships.
    - Create relationships between the elements based on the operator and 
      operands.
    - Store the resulting nodes and relationships into the provided 
      graph_entities.

    :param primitive_expr: The primitive expression to be processed.
    :type primitive_expr: str
    :param graph_entities: A dictionary containing lists of nodes and 
                           relationships.
    :type graph_entities: GraphEntities

    :returns: The result of the evaluation of the primitive expression.
    :rtype: elements.Result

    :raises IndexError: If there is an issue accessing elements from graph_
                        entities.
    :raises ValueError: If the evaluation of the primitive expression fails.

    :example:

        >>> graph_entities = {
        ...     "nodes": [],
        ...     "relationships": []
        ... }
        >>> result = graph_elements_from_primitive_expression("5 + 3", graph_entities)
        >>> print(result.expression)
        '5 + 3'
    """

    elements_list = primitive_expr.split(" ")

    nodes = graph_entities["nodes"]

    left = elements_list[0]
    if left.startswith("R"):
        left = left[1:]
        left = [
            node for node in graph_entities["nodes"]
            if node.id == left
        ][0]
    else:
        left = elements.Number(
            expression=left, 
            value=int(left)
        )
        nodes.append(left)
    
    right = elements_list[2]
    if right.startswith("R"):
        right = right[1:]
        right = [
            node for node in graph_entities["nodes"]
            if node.id == right
        ][0]
    else:
        right = elements.Number(
            expression=right, 
            value=int(right)
        )
        nodes.append(right)
    
    operator = elements.Operator(
        expression=elements_list[1],
        value=elements_list[1]
    )
    nodes.append(operator)

    result = elements.Result(
        expression=primitive_expr, 
        value=eval(f"{left.value} {operator.expression} {right.value}")
    )
    nodes.append(result)

    relationships = graph_entities["relationships"]
    relationships.append(
        elements.IsLeftOperantOf(source=left, target=operator)
    )
    relationships.append(
        elements.IsRightOperantOf(source=right, target=operator)
    )
    relationships.append(
        elements.ResultsIn(source=operator, target=result)
    )

    return result


def graph_from_expression(expr: str) -> GraphEntities:
    """
    Build a graph structure from a given mathematical expression.

    This function iteratively extracts the innermost expressions from the
    input mathematical expression and processes them to construct graph 
    elements. It identifies operations involving addition, subtraction, 
    multiplication, and division, converting them into nodes and relationships
    within the graph.

    The function performs the following steps:
    - Continuously extracts innermost parenthetical expressions.
    - For each innermost expression, it attempts to extract and process 
      the first occurrence of multiplication/division or addition/subtraction.
    - Updates the graph_entities with the created nodes and relationships.
    - Removes redundant parentheses from the original expression as needed.

    :param expr: The mathematical expression to be converted into graph 
                 entities.
    :type expr: str

    :returns: A dictionary containing the constructed graph elements (nodes
              and relationships).
    :rtype: GraphEntities

    :example:

        >>> graph_entities = graph_from_expression("(2 + 3) * (5 - 4)")
        >>> print(graph_entities)
        {
            'nodes': [...],
            'relationships': [...]
        }
    """

    graph_elements = {"nodes": [], "relationships": []}

    expr = remove_redundant_parenthesis(expr)

    while True:
        inner = extract_innermost_parentheses(expr)
        if inner is None:  # No parenthesis found, treat entire expr as inner
            inner = expr
        while True:
            primitive_inner = (
                extract_multiplication_or_division(inner) or
                extract_addition_subtraction(inner)
            )
            if primitive_inner:
                result = graph_elements_from_primitive_expression(
                    primitive_inner,
                    graph_elements
                )
                expr = expr.replace(primitive_inner, f"R{result.id}", 1)
                inner = inner.replace(primitive_inner, f"R{result.id}", 1)
            elif not extract_innermost_parentheses(inner):
                return graph_elements
            else:
                break

            expr = remove_redundant_parenthesis(expr)
