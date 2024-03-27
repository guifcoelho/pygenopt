from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class ConstraintSign(Enum):
    "The available constraint signs"
    LEQ = '<='
    GEQ = '>='
    EQ = '=='

class VarType(Enum):
    "Enumeration of decision varible types"
    BIN = 'bin'
    INT = 'int'
    CNT = 'cnt'

class LinearConstraint:
    "The linear constraint class"
    expression: 'LinearExpression'
    sign: ConstraintSign
    name: Optional[str] = None

    def __init__(self,
                 constr: 'LinearConstraint | tuple[LinearExpression, ConstraintSign]',
                 name: Optional[str] = None):
        if isinstance(constr, LinearConstraint):
            self.expression = constr.expression
            self.sign = constr.sign
        elif isinstance(constr, tuple):
            self.expression, self.sign = constr
        else:
            raise Exception()

        self.name = name


@dataclass
class LinearExpression:
    "A wrapper for general expressions"
    elements: dict[str, float] = field(default_factory=dict, init=False)
    constant: float = field(default=0, init=False)

    def copy(self):
        "Returns a copy of the linear expression"
        new_expr = LinearExpression()
        new_expr.elements = self.elements.copy()
        new_expr.constant = self.constant
        return new_expr

    def _add_expression(self, expr: "LinearExpression", addition: bool = True):
        new_expr = self.copy()
        sign = 1 if addition else -1
        new_expr.constant = self.constant + sign * expr.constant
        for key, val in expr.elements.items():
            new_expr.elements[key] = self.elements.get(key, 0) + sign * val
        return new_expr

    def _add_var(self, var: "Variable", addition: bool = True):
        new_expr = self.copy()
        sign = 1 if addition else -1
        new_expr.elements[var] = self.elements.get(var, 0) + sign
        return new_expr

    def _add_constant(self, val: float, addition: bool = True):
        new_expr = self.copy()
        sign = 1 if addition else -1
        new_expr.constant = self.constant + val * sign
        return new_expr

    def _add(self, other, addition: bool = True):
        if isinstance(other, LinearExpression):
            return self._add_expression(other, addition)
        if isinstance(other, Variable):
            return self._add_var(other, addition)
        return self._add_constant(other, addition)

    def __add__(self, other):
        return self._add(other)

    def __sub__(self, other):
        return self._add(other, False)

    def _multiplication(self, coef: float, multiplication: bool = True):
        new_expr = self.copy()
        coef = float(coef)
        new_expr.elements = {
            key: val * (coef if multiplication else 1/coef)
            for key, val in self.elements.items()
        }
        new_expr.constant *= (coef if multiplication else 1/coef)
        return new_expr

    def __mul__(self, coef: float):
        return self._multiplication(coef)

    def __rmul__(self, coef: float):
        return self._multiplication(coef)

    def __truediv__(self, coef: float):
        return self._multiplication(coef, False)

    def __rtruediv__(self, coef: float):
        return self._multiplication(coef, False)

    def __eq__(self, rhs: 'LinearExpression | Variable | float'):
        return LinearConstraint((LinearExpression() + (self - rhs), ConstraintSign.EQ))

    def __le__(self, rhs: 'LinearExpression | Variable | float'):
        return LinearConstraint((LinearExpression() + (self - rhs), ConstraintSign.LEQ))

    def __ge__(self, rhs: 'LinearExpression | Variable | float'):
        return LinearConstraint((LinearExpression() + (self - rhs), ConstraintSign.GEQ))

    def __neg__(self):
        return LinearExpression() - self

    def __pos__(self):
        return LinearExpression() + self


@dataclass
class Variable:
    "The decicion variable"

    name: str
    column_idx: int = field(default=None, init=False, repr=False)
    vartype: VarType = field(default=VarType.CNT, repr=False)
    lowerbound: float = field(default=None, repr=False)
    upperbound: float = field(default=None, repr=False)

    def __hash__(self):
        return hash(self.name)

    def to_linexpr(self):
        "Transforms variable to linear expression"
        return LinearExpression() + self

    def __add__(self, other: 'LinearExpression | Variable | float'):
        return self.to_linexpr() + other

    def __sub__(self, other: 'LinearExpression | Variable | float'):
        return self.to_linexpr() - other

    def __mul__(self, val: float):
        return self.to_linexpr() * float(val)

    def __rmul__(self, val: float):
        return self * float(val)

    def __truediv__(self, val: float):
        return self.to_linexpr() / float(val)

    def __rtruediv__(self, val: float):
        return self.to_linexpr() / float(val)

    def __eq__(self, rhs: 'LinearExpression | Variable | float'):
        return self.to_linexpr() == rhs

    def __le__(self, rhs: 'LinearExpression | Variable | float'):
        return self.to_linexpr() <= rhs

    def __ge__(self, rhs: 'LinearExpression | Variable | float'):
        return self.to_linexpr() >= rhs

    def __neg__(self):
        return LinearExpression() - self

    def __pos__(self):
        return LinearExpression() + self
