from dataclasses import dataclass, field
from typing import Optional, Any
import time
import random

from pygenopt.enums import ConstraintSign, VarType


@dataclass
class LinearExpression:
    "A wrapper for general expressions"
    elements: dict['Variable', float] = field(default_factory=dict, init=False)
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

    def _add_constant(self, val: float | int, addition: bool = True):
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

    def __radd__(self, other: 'LinearExpression | Variable | float | int'):
        return self + other

    def __sub__(self, other):
        return self._add(other, False)

    def __rsub__(self, other: 'LinearExpression | Variable | float | int'):
        return self - other

    def _multiplication(self, coef: float | int, multiplication: bool = True):
        new_expr = self.copy()
        coef = float(coef)
        new_expr.elements = {
            key: val * (coef if multiplication else 1/coef)
            for key, val in self.elements.items()
        }
        new_expr.constant *= (coef if multiplication else 1/coef)
        return new_expr

    def __mul__(self, coef: float | int):
        return self._multiplication(coef)

    def __rmul__(self, coef: float | int):
        return self._multiplication(coef)

    def __truediv__(self, coef: float | int):
        return self._multiplication(coef, False)

    def __rtruediv__(self, coef: float | int):
        return self._multiplication(coef, False)

    def __eq__(self, rhs: 'LinearExpression | Variable | float | int'):
        rhs_ = rhs.copy() if isinstance(rhs, (LinearExpression, Variable)) else rhs
        constr = LinearConstraint((LinearExpression() + (self.copy() - rhs_), ConstraintSign.EQ))
        constr.expression.constant *= -1
        return constr

    def __le__(self, rhs: 'LinearExpression | Variable | float | int'):
        rhs_ = rhs.copy() if isinstance(rhs, (LinearExpression, Variable)) else rhs
        constr = LinearConstraint((LinearExpression() + (self.copy() - rhs_), ConstraintSign.LEQ))
        constr.expression.constant *= -1
        return constr

    def __ge__(self, rhs: 'LinearExpression | Variable | float | int'):
        rhs_ = rhs.copy() if isinstance(rhs, (LinearExpression, Variable)) else rhs
        constr = LinearConstraint((LinearExpression() + (self.copy() - rhs_), ConstraintSign.GEQ))
        constr.expression.constant *= -1
        return constr

    def __neg__(self):
        return LinearExpression() - self.copy()

    def __pos__(self):
        return LinearExpression() + self.copy()

class LinearConstraint:
    "The linear constraint class"

    expression: 'LinearExpression'
    sign: ConstraintSign
    name: Optional[str] = None
    row: Optional[int] = None
    dual: Optional[float] = None

    _hash: str | None = field(default=None, init=False)
    _default_name: Optional[str] = None

    def __init__(self,
                 constr: 'LinearConstraint | tuple[LinearExpression, ConstraintSign]',
                 name: Optional[str] = None):
        if isinstance(constr, LinearConstraint):
            self.expression = constr.expression
            self.sign = constr.sign
        elif isinstance(constr, tuple):
            self.expression, self.sign = constr
        else:
            raise TypeError(f"The constraint expression must be an inequality, not `{constr}`.")

        self.name = name
        self._hash = hash(f"{time.perf_counter_ns()}{random.random()}")

    def __hash__(self):
        if self._hash is None:
            raise Exception("The hash string of the 'LinearConstraint' object was not set.")
        return self._hash

    @property
    def default_name(self):
        if self._default_name is None:
            raise Exception(
                "The default name of this constraint was not set. "
                "Add this constraint to the problem and run `problem.update()`"
            )
        return self._default_name

    def set_default_name(self, row: int):
        "Sets a default name to be provided to the solver."
        self._default_name = f"__constr{row}"
        return self

    def clear(self):
        "Clears all values from the constraint."
        self._default_name = None
        self.row = None
        self.dual = None

@dataclass
class Variable:
    "The decicion variable"

    name: Optional[str] = field(default=None)
    vartype: VarType = field(default=VarType.CNT)
    lowerbound: Optional[float] = field(default=None)
    upperbound: Optional[float] = field(default=None)
    hotstart: float | None = field(default=None, repr=False)
    target: float | None = field(default=None, repr=False)

    column: int | None = field(default=None, init=False)
    value: float | None = field(default=None, init=False)

    _hash: str | None = field(default=None, init=False, repr=False)
    _default_name: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._hash = hash(f"{time.perf_counter_ns()}{random.random()}")
        if self.vartype == VarType.BIN:
            self.lowerbound = 0
            self.upperbound = 1

        if self.hotstart is not None:
            self._validate_value(self.hotstart, 'hotstart')

        if self.target is not None:
            self._validate_value(self.target, 'target')

    def _validate_value(self, value: float, type) -> None:
        lb = 0 if self.vartype == VarType.BIN else self.lowerbound
        ub = 1 if self.vartype == VarType.BIN else self.upperbound
        if value < lb or value > ub:
            raise ValueError(
                f"The {type} for the decision variable is not within its bounds: "
                f"{value} < {lb}" if value < lb else f"{value} > {ub}"
            )

    @property
    def default_name(self):
        if self._default_name is None:
            raise Exception(
                "The default name of this variable was not set. "
                "Add this variable to the problem and run `problem.update()`"
            )
        return self._default_name

    def __hash__(self):
        if self._hash is None:
            raise Exception("The hash string of the 'Variable' object was not set.")
        return self._hash

    def to_linexpr(self):
        "Transforms the variable into a linear expression"
        return LinearExpression() + self

    def __add__(self, other: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() + other

    def __radd__(self, other: 'LinearExpression | Variable | float | int'):
        return self + other

    def __sub__(self, other: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() - other

    def __rsub__(self, other: 'LinearExpression | Variable | float | int'):
        return self - other

    def __mul__(self, val: float | int):
        return self.to_linexpr() * float(val)

    def __rmul__(self, val: float | int):
        return self * float(val)

    def __truediv__(self, val: float | int):
        return self.to_linexpr() / float(val)

    def __rtruediv__(self, val: float | int):
        return self.to_linexpr() / float(val)

    def __eq__(self, rhs: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() == rhs

    def __le__(self, rhs: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() <= rhs

    def __ge__(self, rhs: 'LinearExpression | Variable | float | int'):
        return self.to_linexpr() >= rhs

    def __neg__(self):
        return LinearExpression() - self

    def __pos__(self):
        return LinearExpression() + self

    def set_default_name(self, column: int):
        "Sets a default name to be provided to the solver."
        self._default_name = f"__var{column}"
        return self

    def set_hotstart(self, value: float):
        "Sets the value to be used as initial solution on the solver."
        value = float(value)
        self._validate_value(value, 'hotstart')
        self.hotstart = float(value)
        return self

    def set_target(self, value: float):
        "Sets the target value for the decision variable."
        value = float(value)
        self._validate_value(value, 'target')
        self.target = value
        return self

    def clear(self):
        "Clears all values from the variable."
        self._default_name = None
        self.value = None
        self.column = None

@dataclass
class ObjectiveFunction:
    expression: "LinearExpression" = field(default_factory=LinearExpression)
    is_minimization: bool = True
    name: str | None = field(default=None)
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.expression += LinearExpression()
