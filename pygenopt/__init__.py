from enum import Enum
from dataclasses import dataclass, field, InitVar
from typing import Optional, Any, Type, Callable
from abc import ABC, abstractmethod
from contextlib import suppress
import time
import random

INF = float('inf')

class ConstraintSign(Enum):
    "The available constraint signs."
    LEQ = '<='
    GEQ = '>='
    EQ = '=='

class VarType(Enum):
    "Enumeration of decision varible types."
    BIN = 'bin'
    INT = 'int'
    CNT = 'cnt'

class SolveStatus(Enum):
    "Enumeration of the solve status."
    OPTIMUM = 'optimum'
    FEASIBLE = 'feasible'
    UNBOUNDED = 'unbounded'
    INFEASIBLE = 'infeasible'
    NOT_SOLVED = 'not_solved'
    UNKNOWN = 'unknown'

pysum = sum

def sum(values):
    if isinstance(values, dict):
        values = list(values.values())
    return LinearExpression() + pysum(values)

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
        constr = LinearConstraint((LinearExpression() + (self - rhs), ConstraintSign.EQ))
        constr.expression.constant *= -1
        return constr

    def __le__(self, rhs: 'LinearExpression | Variable | float | int'):
        constr = LinearConstraint((LinearExpression() + (self - rhs), ConstraintSign.LEQ))
        constr.expression.constant *= -1
        return constr

    def __ge__(self, rhs: 'LinearExpression | Variable | float | int'):
        constr = LinearConstraint((LinearExpression() + (self - rhs), ConstraintSign.GEQ))
        constr.expression.constant *= -1
        return constr

    def __neg__(self):
        return LinearExpression() - self

    def __pos__(self):
        return LinearExpression() + self

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
            raise Exception()

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
    vartype: VarType = field(default=VarType.CNT, repr=False)
    lowerbound: Optional[float] = field(default=None, repr=False)
    upperbound: Optional[float] = field(default=None, repr=False)

    column: int | None = field(default=None, init=False)
    value: float | None = field(default=None, init=False)
    hotstart_value: float | None = field(default=None, init=False)

    _hash: str | None = field(default=None, init=False)
    _default_name: str | None = field(default=None, init=False)

    def __post_init__(self):
        self._hash = hash(f"{time.perf_counter_ns()}{random.random()}")
        if self.vartype == VarType.BIN:
            self.lowerbound = 0
            self.upperbound = 1

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
        self.hotstart_value = value
        return self

    def clear(self):
        "Clears all values from the variable."
        self._default_name = None
        self.value = None
        self.column = None

@dataclass
class ObjectiveFunction:
    name: str | None = field(default=None)
    expression: "LinearExpression" = field(default_factory=LinearExpression)
    is_minimization: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.expression += LinearExpression()

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
    objective_functions: list[ObjectiveFunction] = field(default_factory=list, init=False)

    def __post_init__(self, solver_api: Type['SolverApi']):
        if solver_api is not None:
            self.solver = solver_api()

    @property
    def solve_status(self):
        return self.solver.solve_status or SolveStatus.NOT_SOLVED

    def get_vars_bycolumn(self, columns: list[int]) -> list[Variable]:
        "Finds some variables by searching for column numbers. It will ignore any pending variables."
        return [var for var in self.variables if var.column in columns]

    def get_var_bycolumn(self, column: int) -> Variable:
        """
        Finds a variable by searching for its column number. It will ignore any pending variables.

        It will raise a `KeyError` exception if no variable is found.
        """
        filter_vars = self.get_vars_bycolumn([column])
        if not filter_vars:
            raise KeyError(f"There is no variable with column number '{column}'.")
        return filter_vars[0]

    def get_vars_byname(self, names: list[str]) -> list[Variable]:
        "Finds some variables by searching for a list of names. It will ignore pending ones."
        return [var for var in self.variables if var.name in names or var._default_name in names]

    def get_var_byname(self, name: str) -> Variable:
        """
        Finds a variable by searching for its name. It will ignore pending ones.

        It will raise a `KeyError` exception if no variable is found.
        """
        filter_vars = self.get_vars_byname([name])
        if not filter_vars:
            raise KeyError(f"There is no variable with name '{name}'.")
        return filter_vars[0]

    def get_constrs_byrow(self, rows: list[int]) -> list[LinearConstraint]:
        "Finds some constraints by searching for row numbers. It will ignore pending ones."
        return [constr for constr in self.constraints if constr.row in rows]

    def get_constr_byrow(self, row: int) -> Variable:
        """
        Finds a constraint by searching for its row number. It will ignore pending ones.

        It will raise a `KeyError` exception if no constraint is found.
        """
        filter_constrs = self.get_constrs_byrow([row])
        if not filter_constrs:
            raise KeyError(f"There is no constraint with row number '{row}'.")
        return filter_constrs[0]

    def get_constrs_byname(self, names: list[str]) -> list[LinearConstraint]:
        "Finds some constraints by searching for a list of names. It will ignore pending ones."
        return [constr for constr in self.constraints if constr.name in names or constr._default_name in names]

    def get_constr_byname(self, name: str) -> Variable:
        """
        Finds a contraint by searching for its name. It will ignore pending ones.

        It will raise a `KeyError` exception if no constraint is found.
        """
        filter_constrs = [constr for constr in self.constraints if constr.name == name or constr._default_name == name]
        if not filter_constrs:
            raise KeyError(f"There is no constraint with name '{name}'.")
        return filter_constrs[0]

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

    def add_vars(self, *variables: list | dict | Variable):
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

    def add_constrs(self, *constrs: list | dict | LinearConstraint):
        "Adds some constraints to the model."
        for list_of_constrs in constrs:
            if isinstance(list_of_constrs, dict):
                list_of_constrs = list_of_constrs.values()
            elif isinstance(list_of_constrs, LinearConstraint):
                list_of_constrs = [list_of_constrs]
            self.pending_constraints += list(list_of_constrs)
        return self

    def set_objective(self, objective: ObjectiveFunction | Variable | LinearExpression | float | int):
        "Sets the objetive function to solve for."
        self.objective_functions = list()
        return self.add_objective(objective)

    def add_objectives(self, *objectives: ObjectiveFunction | Variable | LinearExpression | float | int):
        "Adds some objective functions to solve for."
        for objective in objectives:
            self.add_objective(objective)
        return self

    def add_objective(self, objective: ObjectiveFunction | Variable | LinearExpression | float | int):
        "Adds a new objective function to solve for."
        if isinstance(objective, (Variable, float, int)):
            objective += LinearExpression()
        if isinstance(objective, LinearExpression):
            objective = ObjectiveFunction(expression=objective)

        self.objective_functions += [objective]
        return self

    def set_solver(self, solver_api: Type['SolverApi']):
        "Sets the solver from the given interface."
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

        return self

    def set_hotstart(self):
        columns, values = zip(*[
            (var.column, var.hotstart_value)
            for var in self.variables
            if var.hotstart_value is not None
            and var.column is not None
            and (var.lowerbound or -INF) <= var.hotstart_value <= (var.upperbound or INF)
        ])
        self.solver.set_hotstart(columns, values)
        return self

    def solve(self, with_hotstart: bool = False):
        "Updates and runs the solver for the optimization problem."
        if self.solver is None:
            raise Exception("The solver api should be set before solving.")

        self.update()
        if with_hotstart:
            self.set_hotstart()

        if len(self.objective_functions) == 1:
            objective = self.objective_functions[0]
            self.solver.set_objective(objective)
            self.solver.run(objective.options or self.options or dict())
            return self

        def add_constr_callback(constr: LinearConstraint):
            self.add_constr(constr)
            self.update()

        self.solver.run_multiobjective(self.objective_functions, add_constr_callback, self.options or dict())
        return self

    def fetch_solve_status(self):
        self.solver.fetch_solve_status()
        return self

    def fetch_solution(self):
        "Retrieve all solution values after a solve."
        self.solver.fetch_solve_status()
        if self.solve_status in [SolveStatus.FEASIBLE, SolveStatus.OPTIMUM]:
            self.solver.fetch_solution()
            for variable in self.variables:
                variable.value = self.solver.get_solution(variable)
        return self

    def get_objectivefunction_value(self) -> float:
        return self.solver.get_objective_value()

    def fetch_duals(self):
        "Retrieve all dual values after a solve."
        self.solver.fetch_duals()
        for constr in self.constraints:
            constr.dual = self.solver.get_dual(constr)
        return self

    def get_solution(self, variable: Variable) -> float:
        "Returns the solution value of a variable."
        return self.solver.get_solution(variable)

    def get_dual(self, constraint: LinearConstraint) -> float:
        "Returns the dual value of a constraint."
        return self.solver.get_dual(constraint)

    def clear(self):
        "Returns a fresh instance of the optimization problem."
        if self.solver is not None:
            self.solver.clear()
        return Problem(name=self.name, solver=self.solver, options=self.options)

    def clear_solver(self):
        """
        Clears the solver but keeps the optimization problem.
        Running `update()` will rebuild the solver model with the previous
        variables, constraints and objective function.
        """
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

    def to_mps(self, path: str) -> None:
        "Exports the model to an MPS file. It will ignore pending variables and constraints."
        self.solver.to_mps(path)


@dataclass
class SolverApi(ABC):
    solver_name: str = field(default=None, init=False)
    model: Any | None = field(default=None, init=False, repr=False)
    solve_status: SolveStatus | None = field(default=None, init=False)
    solution: list | dict | None = field(default=None, init=False, repr=False)
    duals: list | dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.init_model()

    @abstractmethod
    def init_model(self) -> 'SolverApi':
        "Initializes the solver."
        ...

    @abstractmethod
    def add_var(self, variable: Variable) -> 'SolverApi':
        "Adds a variable to the solver."
        ...

    def add_vars(self, variables: list[Variable]) -> 'SolverApi':
        "Adds some variables to the solver."
        for var in variables:
            self.add_var(var)
        return self

    @abstractmethod
    def del_var(self, variable: Variable) -> 'SolverApi':
        "Deletes the whole column from the actual optimization model."
        ...

    def del_vars(self, variables: list[Variable]) -> 'SolverApi':
        "Deletes whole columns from the actual optimization model."
        for var in variables:
            self.del_var(var)
        return self

    @abstractmethod
    def add_constr(self, constraint: LinearConstraint) -> 'SolverApi':
        "Adds a contraint to the solver."
        ...

    def add_constrs(self, constrs: list[LinearConstraint]) -> 'SolverApi':
        "Adds some constraints to the solver."
        for constr in constrs:
            self.add_constr(constr)
        return self

    @abstractmethod
    def set_objective(self, objetive_function: ObjectiveFunction | Variable | LinearExpression | float | int) -> "SolverApi":
        "Sets the problem objective function to the solver."
        ...

    @abstractmethod
    def set_option(self, name: str, value) -> 'SolverApi':
        "Sets an option to the solver."
        ...

    def set_options(self, options: dict[str, Any]) -> 'SolverApi':
        "Sets some options to the solver."
        for name, val in options.items():
            self.set_option(name, val)
        return self

    @abstractmethod
    def get_option(self, name: str) -> Any:
        "Returns the value of an option from the solver."
        ...

    def get_options(self, options: list[str]) -> dict[str, Any]:
        "Returns the values of some options from the solver."
        return {
            option: self.get_option(option)
            for option in options
        }

    @abstractmethod
    def fetch_solution(self) -> 'SolverApi':
        "Retrieve all solution values after a solve."
        ...

    @abstractmethod
    def fetch_duals(self) -> 'SolverApi':
        "Retrieves all dual values after a solve."
        ...

    @abstractmethod
    def get_objective_value(self) -> float:
        "Returns the model's objective function value."
        ...

    @abstractmethod
    def get_solution(self, variable: Variable) -> float:
        "Returns the solution value of a variable."
        ...

    @abstractmethod
    def get_dual(self, constraint: LinearConstraint) -> float:
        "Returns the dual value of a constraint."
        ...

    @abstractmethod
    def fetch_solve_status(self) -> 'SolverApi':
        "Sets the status of the solving process"
        ...

    @abstractmethod
    def set_hotstart(self, columns: list[int], values: list[float]) -> 'SolverApi':
        "Provides the solver with an initial solution (even if it is partial one)."
        ...

    @abstractmethod
    def run(self, options: Optional[dict[str, Any]] = None) -> 'SolverApi':
        "Runs the solver for the optimization problem with a single objective."
        ...

    @abstractmethod
    def run_multiobjective(self,
                           objectives: list[ObjectiveFunction],
                           add_constr_callback: Callable[[LinearConstraint], None],
                           options: Optional[dict[str, Any]] = None) -> 'SolverApi':
        """
        Runs the solver for the optimization problem with multiples objectives.

        The callback function must be run for every new constraint added to the model.
        """
        ...

    def clear(self) -> 'SolverApi':
        "Clears the model."
        self.solution.clear()
        self.duals.clear()
        self.init_model()
        return self

    @abstractmethod
    def to_mps(self, path: str) -> None:
        "Exports the model to file in the MPS format."
