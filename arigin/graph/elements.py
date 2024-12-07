import random
import pandas as pd
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Any, Union, List


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
    id: str = Field(default_factory=lambda: "%032x" % random.getrandbits(128))


class Node(AbstractModel):
    """
    Object describing a node in a graph.
    """
    type: Optional[Enum] = Field(
        None, 
        description=
            "Attribute to discriminate between different "
            "categorical types of the same node class."
    )
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
    value: float = Field(description="Float value representing a number")


class LeftOperand(Number):
    """
    Representation of a left operant.
    """
    pass


class RightOperand(Number):
    """
    Representation of a right operant.
    """
    pass


class Operator(Node):
    """
    Representation of arithmetic operator.
    """
    type: OperatorType = Field(
        None, 
        description="Arithmetic operators can have different types."
    )


class Result(Node):
    """
    Representation of a result of an expression.
    """
    value: Optional[float] = Field(
        None, 
        description=
            "Value representing a result. Can also be Null "
            "if the result is unknown."
        )


class IsLeftOperantOf(Relationship):
    """
    Representing the relation between a number beeing the left operant
    to an operator.
    """
    source: Union[LeftOperand, Result]
    target: Operator


class IsRightOperantOf(Relationship):
    """
    Representing the relation between a number beeing the right operant
    to an operator.
    """
    source: Union[RightOperand, Result]
    target: Operator


class ResultsIn(Relationship):
    """
    Representing the relation between an operator and a result.
    """
    source: Operator
    target: Result


def model_to_frame(model: Union[AbstractModel, List[AbstractModel]]):
    """
    Method to convert a (list of) model to a pandas.DataFrame including the
    actual class name and the model's attributes.
    """

    if not isinstance(model, list):
        model = [model]

    df = pd.DataFrame([single.model_dump() for single in model])
    df["class"] = [single.__class__.__name__ for single in model]
    return df


def join_categorical(df: pd.DataFrame, columns=list()):
    """
    Join categorical columns and make one output column of it, while
    dropping the original ones.
    """

    outname = "_".join(columns)
    df.fillna(dict.fromkeys(columns, "none"), inplace=True)
    df[outname] = df[columns].apply("_".join, axis=1)
    return df


def mask_result_values(df: pd.DataFrame):
    """
    Mask any value column of a Result Node class.
    """
    df = df.copy()  # Masking not to overwrite the original value
    df.loc[df['class'] == 'Result', 'value'] = None
    return df
