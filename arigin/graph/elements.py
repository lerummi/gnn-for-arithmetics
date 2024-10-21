import random
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any, Union


class OperatorType(str, Enum):
    MULTIPLICATION = "*"
    DIVISION = "/"
    ADDITION = "+"
    SUBTRACTION = "-"


class RelationshipType(str, Enum):
    IS_LEFT_OPERANT_OF = "is_left_operant_of"
    IF_RIGHT_OPERANT_OF = "is_right_operant_of"
    RESULTS_IN = "results_in"


class AbstractModel(BaseModel):
    """
    Object acting as abstract base model for any graph element.
    """
    #type: str = Field(default_factory=str)
    id: str = Field(default_factory=lambda: "%032x" % random.getrandbits(128))

    #@model_validator(mode="before")
    #def set_type(cls, values):
    #    # Set type to the name of the subclass dynamically
    #    values['type'] = cls.__name__
    #    return values



class Node(AbstractModel):
    """
    Object describing a node in a graph.
    """
    value: Optional[Any] = None
    expression: str = Field(
        description=
            "String expression fundamental for definition of the "
            "Node. If Number / Operator, then string representing "
            "the number / operator. If Result, then the expression "
            "to be evaluated."
    )


class Relationship(AbstractModel):
    """
    Object describing a relationship in a graph.
    """
    source: Node
    target: Node


class Number(Node):
    """
    Representation of a number.
    """
    value: int = Field(description="Integer value representing a number")


class Operator(Node):
    """
    Representation of arithmetic operator.
    """
    value: OperatorType = Field(description="Type of arithmetic operator")


class Result(Node):
    """
    Representation of a result of an expression.
    """
    pass


class IsLeftOperantOf(Relationship):
    """
    Representing the relation between a number beeing the left operant
    to an operator.
    """
    source: Union[Number, Result]
    target: Operator


class IsRightOperantOf(Relationship):
    """
    Representing the relation between a number beeing the right operant
    to an operator.
    """
    source: Union[Number, Result]
    target: Operator


class ResultsIn(Relationship):
    """
    Representing the relation between an operator and a result.
    """
    source: Operator
    target: Result
