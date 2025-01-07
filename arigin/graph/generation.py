import re
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Union, Optional, Tuple
from arigin.graph import elements
from arigin.expressions import generate


# Placeholder for storing graph entities, i.e. dictionary of nodes
# and relationships
GraphEntities = Dict[
    str, 
    List[Union[elements.Node, elements.Relationship]]
    ]


MATCH_PATTERN = r"\w*\.?\w*"


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
        '[LeftOperant] [Operator] [RightOperant]' 
    """
    return extract_first_pattern(
        rf"{MATCH_PATTERN}\s*[*/]\s*{MATCH_PATTERN}", 
        expression
    )


def extract_addition_subtraction(expression: str) -> Optional[str]:
    """
    Given a expression, extract the first part connected to
    addition and subtraction. Returns primitive expression of the 
    form
        '[LeftOperant] [Operator] [RightOperant]' 
    """
    return extract_first_pattern(
        rf"{MATCH_PATTERN}\s*[+-]\s*{MATCH_PATTERN}",
        expression
    )


def remove_redundant_parenthesis(expression: str) -> str:
    """
    Replace patterns like '( OPERATOR<alphanumeric> )' -> 'OPERATOR<alphanumeric>' and
    '( <float> )' -> '<float>'  and so remove redundant parentheses. 
    This also handles cases where there are multiple layers of redundant
    parentheses around OPERATOR<alphanumeric>.

    :param expression: The mathematical expression as a string.
    :type expression: str
    :return: The cleaned expression with redundant parentheses removed.
    :rtype: str
    """

    # Regular expression to match redundant parentheses around R<alphanumeric>
    patterns = [rf"\(\s*({MATCH_PATTERN})\s*\)"]

    # Continuously remove redundant parentheses as long as they exist
    for pattern in patterns:
        while re.search(pattern, expression):
            expression = re.sub(pattern, r'\1', expression)

    return expression


def graph_elements_from_primitive_expression(
        primitive_expr: str,
        graph_entities: GraphEntities) -> elements.Operator:
    """
    Create graph elements from a given primitive expression and update the 
    graph entities.

    A primitive expression is expected to be in the form:
    '[LeftOperant] [Operator] [RightOperant]' or
    'OPERATOR<alphanumeric> [Operator] OPERATOR<alphanumeric>' or a mixture of both, 
    where 'OPERATOR' represents a relationship to a node in the graph.

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

    :returns: The operator of the evaluation of the primitive expression.
    :rtype: elements.Operator

    :raises IndexError: If there is an issue accessing elements from graph_
                        entities.
    :raises ValueError: If the evaluation of the primitive expression fails.

    :example:

        >>> graph_entities = {
        ...     "nodes": [],
        ...     "relationships": []
        ... }
        >>> operator = graph_elements_from_primitive_expression("5 + 3", graph_entities)
    """

    elements_list = primitive_expr.split(" ")

    nodes = graph_entities["nodes"]

    left = elements_list[0]
    if left.isalnum():
        left = elements.node_by_id(nodes, left)
    else:
        left = elements.LeftOperand(
            expression=left, 
            value=left
        )
        nodes.append(left)
    
    right = elements_list[2]
    if right.isalnum():
        right = elements.node_by_id(nodes, right)
    else:
        right = elements.RightOperand(
            expression=right, 
            value=right
        )
        nodes.append(right)
    
    operator = elements.Operator(
        expression=elements_list[1],
        type=elements_list[1]
    )
    nodes.append(operator)

    relationships = graph_entities["relationships"]
    relationships.append(
        elements.IsLeftOperantOf(source=left, target=operator)
    )
    relationships.append(
        elements.IsRightOperantOf(source=right, target=operator)
    )

    return operator


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

    graph_entities = {"nodes": [], "relationships": []}

    try:
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
                    operator = graph_elements_from_primitive_expression(
                        primitive_inner,
                        graph_entities
                    )
                    expr = expr.replace(primitive_inner, operator.id, 1)
                    inner = inner.replace(primitive_inner, operator.id, 1)
                elif not extract_innermost_parentheses(inner):
                    return graph_entities
                else:
                    break

                expr = remove_redundant_parenthesis(expr)
    except ZeroDivisionError:
        return graph_entities


def generate_multiple_graphs(
        n_graphs=1000,
        min_numbers=2,
        max_numbers=4
) -> Tuple[GraphEntities, np.ndarray]:
    """
    Generate multiple graphs with random mathematical expressions.

    This function generates multiple graphs with random mathematical 
    expressions, where each graph consists of nodes and relationships 
    representing the expression. The number of graphs to generate, as well 
    as the minimum and maximum number of numbers in each expression, can be 
    specified.

    :param n_graphs: The number of graphs to generate.
    :type n_graphs: int
    :param min_numbers: The minimum number of numbers in each expression.
    :type min_numbers: int
    :param max_numbers: The maximum number of numbers in each expression.
    :type max_numbers: int

    :returns: GraphEntities and results of the generated graphs.
    :rtype: Tuple[GraphEntities, np.ndarray]

    :example:

        >>> graphs = generate_multiple_graphs(n_graphs=10, min_numbers=2, max_numbers=4)
        >>> print(len(graphs))
        10
    """

    graph_entities = {"nodes": [], "relationships": []}
    results = []
    batch = []
    graph_i = 0
    for _ in tqdm(range(n_graphs), total=n_graphs):
        expr = generate(min_numbers, max_numbers)
        single_graph_entities = graph_from_expression(expr)
        try:
            y = eval(expr)
        except ZeroDivisionError:
            continue
        n_nodes = len(single_graph_entities["nodes"])
        graph_entities["nodes"] += single_graph_entities["nodes"]
        graph_entities["relationships"] += single_graph_entities["relationships"]
        batch += [graph_i] * n_nodes

        results.append(y)
        graph_i += 1

    graph_entities.update({"batch": batch})
    results = np.array(results).reshape(-1, 1)

    return graph_entities, results
