from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any, Type
from abc import ABC, abstractmethod

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
    row: Optional[int] = None
    dual: Optional[float] = None

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
    vartype: VarType = field(default=VarType.CNT, repr=False)
    lowerbound: float = field(default=None, repr=False)
    upperbound: float = field(default=None, repr=False)
    column: int = field(default=None, init=False)
    value: float = field(default=None, init=False)

    def __post_init__(self):
        if self.vartype == VarType.BIN:
            self.lowerbound = 0
            self.upperbound = 1

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

@dataclass
class Model:
    "The optimization model class"

    name: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    variables: list[Variable] = field(default_factory=list, init=False)
    constraints: list[LinearConstraint] = field(default_factory=list, init=False)
    objective_function: LinearExpression = field(default=None, init=False)
    is_minimization: bool = field(default=True, init=False)
    solver: 'SolverApi' = field(default=None, init=False)

    def set_option(self, name: str, value):
        "Sets the solver option"
        self.options[name] = value
        return self

    def set_options(self, options: dict[str, Any]):
        "Sets a some solver options"
        for key, val in options.items():
            self.set_option(key, val)
        return self

    def add_var(self, variable: Variable):
        "Adds a new column to the optimization model."
        self.variables += [variable]
        return self

    def add_vars(self, *variables):
        "Adds some columns to the optimization model."
        if len(variables) == 1 and isinstance(variables[0], (list, tuple)):
            variables = variables[0]

        for variable in variables:
            self.add_var(variable)

        return self

    def add_constr(self, constr: LinearConstraint):
        "Adds a new constraint to the model."
        self.constraints += [constr]
        return self

    def add_constrs(self, *constrs):
        "Adds some constraints to the model."
        if len(constrs) == 1 and isinstance(constrs[0], (list, tuple)):
            constrs = constrs[0]
        self.constraints += list(constrs)
        return self

    def set_objective(self, objetive_function: LinearExpression, is_minimization = True):
        "Sets the objetive function to solve for"
        self.objective_function = LinearExpression() + objetive_function
        self.is_minimization = is_minimization
        return self

    def set_solver(self, solver: Type['SolverApi']):
        self.solver = solver()
        return self

    def build_model(self) -> 'Model':
        for idx, variable in enumerate(self.variables):
            variable.column = idx
        self.solver.add_vars(self.variables)

        for idx, constr in enumerate(self.constraints):
            constr.row = idx
        self.solver.add_constrs(self.constraints)

        self.solver.set_objective(self.objective_function, self.is_minimization)

        return self

    def run(self, solver: 'SolverApi') -> 'Model':
        "Runs the solver for the optimization problem."
        self.set_solver(solver).build_model()
        self.solver.run(self.options)
        return self

    def fetch_solution(self) -> 'Model':
        self.solver.fetch_solution()
        for variable in self.variables:
            variable.value = self.solver.get_solution(variable)
        return self

    def fetch_duals(self) -> 'Model':
        self.solver.fetch_duals()
        for constr in self.constraints:
            constr.dual = self.solver.get_dual(constr)
        return self

    def get_solution(self, variable: Variable) -> float | None:
        return self.solver.get_solution(variable)

    def get_dual(self, constraint: LinearConstraint) -> float | None:
        return self.solver.get_dual(constraint)

@dataclass
class SolverApi(ABC):
    solver_name: str = field(default=None, init=False)
    model: Any | None = field(default=None, init=False)
    solution: dict[Variable, float] | None = field(default=None, init=False)
    duals: dict[LinearConstraint, float] | None = field(default=None, init=False)

    def __post_init__(self):
        self.init_model()

    @abstractmethod
    def init_model(self) -> 'SolverApi':
        ...

    @abstractmethod
    def add_var(self, variable: Variable) -> 'SolverApi':
        ...

    @abstractmethod
    def add_vars(self, variables: list[Variable]) -> 'SolverApi':
        ...

    @abstractmethod
    def add_constr(self, constraint: LinearConstraint) -> 'SolverApi':
        ...

    @abstractmethod
    def add_constrs(self, constraint: list[LinearConstraint]) -> 'SolverApi':
        ...

    @abstractmethod
    def set_objective(self, expr: LinearExpression, is_minimization: bool = True) -> 'SolverApi':
        ...

    @abstractmethod
    def set_option(self, name: str, value) -> 'SolverApi':
        ...

    def set_options(self, options: dict[str, Any]) -> 'SolverApi':
        for name, val in options.items():
            self.set_option(name, val)
        return self

    @abstractmethod
    def fetch_solution(self) -> 'SolverApi':
        ...

    @abstractmethod
    def fetch_duals(self) -> 'SolverApi':
        ...

    @abstractmethod
    def get_solution(self, variable: Variable) -> float:
        ...

    @abstractmethod
    def get_dual(self, constraint: LinearConstraint) -> float:
        ...

    def set_solution(self, solution: dict[Variable, float]) -> 'SolverApi':
        self.solution = solution
        return self

    def set_duals(self, duals: dict[LinearConstraint, float]) -> 'SolverApi':
        self.duals = duals
        return self

    @abstractmethod
    def run(self, options: Optional[dict[str, Any]] = None) -> 'SolverApi':
        ...
