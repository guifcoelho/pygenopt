from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
        "Deletes the whole column from the actual optimization model."
        ...

    def del_vars(self, variables: list[Variable]) -> "AbstractSolverApi":
        "Deletes whole columns from the actual optimization model."
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
        "Retrieve all solution values after a solve."
        ...

    @abstractmethod
    def fetch_duals(self) -> "AbstractSolverApi":
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
    def fetch_solve_status(self) -> "AbstractSolverApi":
        "Sets the status of the solving process"
        ...

    @abstractmethod
    def set_hotstart(self, columns: list[int], values: list[float]) -> "AbstractSolverApi":
        "Provides the solver with an initial solution (even if it is partial one)."
        ...

    @abstractmethod
    def run(self, options: Optional[dict[str, Any]] = None) -> "AbstractSolverApi":
        "Runs the solver for the optimization problem with a single objective."
        ...

    def run_multiobjective(self,
                           objectives: list[ObjectiveFunction],
                           add_constr_callback: Callable[[LinearConstraint], None],
                           options: Optional[dict[str, Any]] = None) -> "AbstractSolverApi":
        """
        Runs the solver for the optimization problem with multiples objectives.

        The callback function must be run for every new constraint added to the model.
        """
        for idx, objective in enumerate(objectives):
            if self.show_log:
                if idx > 0:
                    print()
                obj_name = f"'{objective.name}' " if objective.name is not None else ""
                print(
                    f">> Solving for objective {obj_name}({idx+1} of {len(objectives)}, "
                    f"sense: {'Minimization' if objective.is_minimization else 'Maximization'})"
                )

            self.set_objective(objective)
            self.run(objective.options or options)
            self.fetch_solve_status()

            if self.solve_status in [SolveStatus.FEASIBLE, SolveStatus.OPTIMUM] and idx < len(objectives) - 1:
                self.fetch_solution()
                self.set_hotstart(list(range(len(self.solution))), self.solution)

                add_constr_callback(objective.expression == self.get_objective_value())
            else:
                break

        return self

    def clear(self) -> "AbstractSolverApi":
        "Clears the model."
        self.solution.clear()
        self.duals.clear()
        self.init_model()
        return self

    @abstractmethod
    def to_mps(self, path: str) -> None:
        "Exports the model to file in the MPS format."
