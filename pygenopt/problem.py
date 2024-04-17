from dataclasses import dataclass, field, InitVar
from typing import Any, Type
from contextlib import suppress

from pygenopt import Variable, LinearConstraint, ObjectiveFunction, SolveStatus, LinearExpression
from pygenopt.abstractsolverapi import SolverApi
from pygenopt.constants import INF


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

    def solve(self, with_hotstart: bool = False, fetch_solution: bool = True, fetch_duals: bool = False):
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

        else:
            def add_constr_callback(constr: LinearConstraint):
                self.add_constr(constr)
                self.update()

            self.solver.run_multiobjective(self.objective_functions, add_constr_callback, self.options or dict())

        if fetch_solution:
            self.fetch_solution()

        if fetch_duals:
            self.fetch_duals()

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
