from enum import Enum
from dataclasses import dataclass, field, InitVar
from typing import Optional, Any, Type
from abc import ABC, abstractmethod
from contextlib import suppress


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

class SolveStatus(Enum):
    OPTIMUM = 'optimum'
    FEASIBLE = 'feasible'
    UNBOUNDED = 'unbounded'
    INFEASIBLE = 'infeasible'
    NOT_SOLVED = 'not_solved'
    UNKNOWN = 'unknown'

class LinearConstraint:
    "The linear constraint class"
    expression: 'LinearExpression'
    sign: ConstraintSign
    name: Optional[str] = None
    default_name: Optional[str] = None
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

    def set_default_name(self, row: int):
        self.default_name = f"__constr{row}"
        return self

    def clear(self):
        self.default_name = None
        self.row = None
        self.dual = None


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

    def __radd__(self, other: 'LinearExpression | Variable | float'):
        return self + other

    def __sub__(self, other):
        return self._add(other, False)

    def __rsub__(self, other: 'LinearExpression | Variable | float'):
        return self - other

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
    default_name: Optional[str] = field(default=None, init=False)
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

    def __radd__(self, other: 'LinearExpression | Variable | float'):
        return self + other

    def __sub__(self, other: 'LinearExpression | Variable | float'):
        return self.to_linexpr() - other

    def __rsub__(self, other: 'LinearExpression | Variable | float'):
        return self - other

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

    def set_default_name(self, column: int):
        self.default_name = f"__var{column}"
        return self

    def clear(self):
        self.default_name = None
        self.value = None
        self.column = None

@dataclass
class Problem:
    "The optimization problem class"

    name: str = None
    solver_api: InitVar[Type['SolverApi']] = None
    options: dict[str, Any] = field(default_factory=dict)

    solver: 'SolverApi' = field(default=None, init=False)
    variables: list[Variable] = field(default_factory=list, init=False)
    pending_variables: list[Variable] = field(default_factory=list, init=False)
    deleting_variables: list[Variable] = field(default_factory=list, init=False)
    constraints: list[LinearConstraint] = field(default_factory=list, init=False)
    pending_constraints: list[LinearConstraint] = field(default_factory=list, init=False)
    objective_function: LinearExpression = field(default=None, init=False)
    is_minimization: bool = field(default=True, init=False)

    def __post_init__(self, solver_api: Type['SolverApi']):
        if solver_api is not None:
            self.solver = solver_api()

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
        self.pending_variables += [variable]
        return self

    def add_vars(self, *variables):
        "Adds some columns to the optimization model."
        for list_of_variables in variables:
            if isinstance(list_of_variables, dict):
                list_of_variables = list_of_variables.values()
            elif isinstance(list_of_variables, Variable):
                list_of_variables = [list_of_variables]
            self.pending_variables += list(list_of_variables)
        return self

    def del_var(self, variable: Variable | int):
        """
        Marks a variable (object ou column index) to be deleted from the model.
        Run `update_model()` before solving.
        """
        if isinstance(variable, int):
            try:
                variable = [
                    var
                    for var in self.variables
                    if var.column == variable
                ][0]
            except Exception:
                raise Exception(f"No variable with column index {variable} was added to the model.")

            self.deleting_variables += [variable]
            return self

        if isinstance(variable, Variable):
            if variable in self.variables:
                self.deleting_variables += [variable]
                with suppress(ValueError):
                    self.pending_variables.remove(variable)
                return self

    def del_vars(self, *variables):
        """
        Marks a set of variables (objects ou column indexes) to be deleted from the model.
        Run `update_model()` before solving.
        """
        if len(variables) == 1 and isinstance(variables[0], (list, tuple)):
            variables = variables[0]
        for variable in variables:
            self.del_var(variable)
        return self

    def add_constr(self, constr: LinearConstraint):
        "Adds a new constraint to the model."
        self.pending_constraints += [constr]
        return self

    def add_constrs(self, *constrs: tuple):
        "Adds some constraints to the model."
        for list_of_constrs in constrs:
            if isinstance(list_of_constrs, dict):
                list_of_constrs = list_of_constrs.values()
            elif isinstance(list_of_constrs, LinearConstraint):
                list_of_constrs = [list_of_constrs]
            self.pending_constraints += list(list_of_constrs)
        return self

    def set_objective(self, objetive_function: LinearExpression, is_minimization = True):
        "Sets the objetive function to solve for"
        self.objective_function = LinearExpression() + objetive_function
        self.is_minimization = is_minimization
        return self

    def set_solver(self, solver_api: Type['SolverApi']):
        self.solver = solver_api()
        return self

    def update(self):
        """
        Deletes and adds any pending variables and constraints to the model,
        and sets the objective function.
        """
        for deleting_var in self.deleting_variables:
            self.solver.del_var(deleting_var)  # model should delete the whole column
            for idx in range(deleting_var.column + 1, len(self.variables)):
                self.variables[idx].column -= 1
            self.variables.remove(deleting_var)
            for constr_list in [self.constraints, self.pending_constraints]:
                for constr in constr_list:
                    with suppress(KeyError):
                        constr.expression.elements.pop(deleting_var)
        self.deleting_variables = []

        for idx, variable in enumerate(self.pending_variables):
            variable.column = idx + len(self.variables)
            variable.set_default_name(variable.column)
        self.solver.add_vars(self.pending_variables)
        self.variables += self.pending_variables
        self.pending_variables = []

        for idx, constr in enumerate(self.pending_constraints):
            constr.row = idx + len(self.constraints)
            constr.set_default_name(constr.row)
        self.solver.add_constrs(self.pending_constraints)
        self.constraints += self.pending_constraints
        self.pending_constraints = []

        self.solver.set_objective(self.objective_function, self.is_minimization)

        return self

    def run(self, with_hot_start: bool = False):
        "Updates and runs the solver for the optimization problem."
        if self.solver is None:
            raise Exception("The solver api should be set before solving.")
        self.update()
        if with_hot_start:
            self.solver.set_hotstart(self.variables)
        self.solver.run(self.options)
        return self

    def fetch_solution(self):
        if self.solve_status in [SolveStatus.FEASIBLE, SolveStatus.OPTIMUM]:
            self.solver.fetch_solution()
            for variable in self.variables:
                variable.value = self.solver.get_solution(variable)
        return self

    def fetch_duals(self):
        self.solver.fetch_duals()
        for constr in self.constraints:
            constr.dual = self.solver.get_dual(constr)
        return self

    def get_solution(self, variable: Variable) -> float | None:
        return self.solver.get_solution(variable)

    def get_dual(self, constraint: LinearConstraint) -> float | None:
        return self.solver.get_dual(constraint)

    def clear(self):
        if self.solver is not None:
            self.solver.clear()
        return Problem(name=self.name, solver=self.solver, options=self.options)

    def clear_model(self):
        self.pending_variables = self.variables + self.pending_variables
        for var in self.pending_variables:
            var.clear()
        self.variables = []

        self.pending_constraints = self.constraints + self.pending_constraints
        for constr in self.pending_constraints:
            constr.clear()
        self.constraints = []

        if self.solver is not None:
            self.solver.clear()
        return self

    @property
    def solve_status(self):
        return self.solver.solve_status or SolveStatus.NOT_SOLVED

@dataclass
class SolverApi(ABC):
    solver_name: str = field(default=None, init=False)
    model: Any | None = field(default=None, init=False)
    solve_status: SolveStatus | None = field(default=None, init=False)
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
    def del_var(self, variable: Variable) -> 'SolverApi':
        "Deletes the whole column from the actual optimization model."
        ...

    @abstractmethod
    def del_vars(self, variables: list[Variable]) -> 'SolverApi':
        "Deletes whole columns from the actual optimization model."
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

    @abstractmethod
    def fetch_solve_status(self) -> 'SolverApi':
        ...

    @abstractmethod
    def set_hotstart(self, variables: list[Variable]) -> 'SolverApi':
        ...

    @abstractmethod
    def run(self, options: Optional[dict[str, Any]] = None) -> 'SolverApi':
        ...

    def clear(self) -> 'SolverApi':
        self.solution.clear()
        self.duals.clear()
        return self