from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from pygenopt import SolveStatus, Variable, LinearConstraint, ObjectiveFunction, LinearExpression


@dataclass
class AbstractSolverApi(ABC):
    solver_name: str = field(default=None, init=False)
    model: Any | None = field(default=None, init=False, repr=False)
    solve_status: SolveStatus | None = field(default=None, init=False)
    solution: list | dict | None = field(default=None, init=False, repr=False)
    duals: list | dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.init_model()

    @property
    @abstractmethod
    def show_log(self):
        ...

    @abstractmethod
    def init_model(self) -> "AbstractSolverApi":
        "Initializes the solver."
        ...

    @abstractmethod
    def add_var(self, variable: Variable) -> "AbstractSolverApi":
        "Adds a variable to the solver."
        ...

    def add_vars(self, variables: list[Variable]) -> "AbstractSolverApi":
        "Adds some variables to the solver."
        for var in variables:
            self.add_var(var)
        return self

    @abstractmethod
    def del_var(self, variable: Variable) -> "AbstractSolverApi":
        "Deletes a column from the actual optimization model."
        ...

    def del_vars(self, variables: list[Variable]) -> "AbstractSolverApi":
        "Deletes some columns from the actual optimization model."
        for var in variables:
            self.del_var(var)
        return self

    @abstractmethod
    def add_constr(self, constraint: LinearConstraint) -> "AbstractSolverApi":
        "Adds a contraint to the solver."
        ...

    def add_constrs(self, constrs: list[LinearConstraint]) -> "AbstractSolverApi":
        "Adds some constraints to the solver."
        for constr in constrs:
            self.add_constr(constr)
        return self

    @abstractmethod
    def del_constr(self, constraint: LinearConstraint) -> "AbstractSolverApi":
        "Deletes a row from the actual optimization model."
        ...

    def del_constrs(self, constraints: list[LinearConstraint]) -> "AbstractSolverApi":
        "Deletes some rows from the actual optimization model."
        for constr in constraints:
            self.del_constr(constr)
        return self

    @abstractmethod
    def set_objective(self, objetive_function: ObjectiveFunction | Variable | LinearExpression | float | int) -> "AbstractSolverApi":
        "Sets the problem objective function to the solver."
        ...

    @abstractmethod
    def set_option(self, name: str, value) -> "AbstractSolverApi":
        "Sets an option to the solver."
        ...

    def set_options(self, options: dict[str, Any]) -> "AbstractSolverApi":
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
    def fetch_solution(self) -> "AbstractSolverApi":
        "Retrieve all variable and dual values after a solve."
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
    def fetch_solve_status(self) -> "AbstractSolverApi":
        "Sets the status of the solving process"
        ...

    @abstractmethod
    def set_hotstart(self, columns: list[int], values: list[float]) -> "AbstractSolverApi":
        "Provides the solver with an initial solution (even if it is partial one)."
        ...

    @abstractmethod
    def run(self, options: Optional[dict[str, Any]] = None) -> "AbstractSolverApi":
        "Runs the solver for the optimization problem."
        ...

    def clear(self) -> "AbstractSolverApi":
        "Clears the model."
        if self.solution is not None:
            self.solution.clear()
        if self.duals is not None:
            self.duals.clear()
        self.init_model()
        return self

    @abstractmethod
    def pull_from_model(self) -> tuple[list[Variable], list[LinearConstraint], ObjectiveFunction]:
        "Pulls the variable, constraints and objective function from the model."
        ...