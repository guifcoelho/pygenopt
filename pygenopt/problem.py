from dataclasses import dataclass, field, InitVar
from typing import Any, Type, Optional
from contextlib import suppress

from pygenopt import Variable, LinearConstraint, ObjectiveFunction, SolveStatus, LinearExpression, VarType
from pygenopt.solvers.abstractsolverapi import AbstractSolverApi
from pygenopt.solvers import HighsApi
from pygenopt.constants import INF, TOL


@dataclass
class Problem:
    "The optimization problem class"

    name: Optional[str] = field(default=None)
    solver_api: InitVar[Type["AbstractSolverApi"]] = field(default=None)
    options: dict[str, Any] = field(default_factory=dict)
    decimal_tol: float = field(default=TOL)

    solver: "AbstractSolverApi" = field(default=None, init=False)
    variables: list[Variable] = field(default_factory=list, init=False)
    pending_variables: list[Variable] = field(default_factory=list, init=False)
    deleting_variables: list[Variable] = field(default_factory=list, init=False)
    constraints: list[LinearConstraint] = field(default_factory=list, init=False)
    pending_constraints: list[LinearConstraint] = field(default_factory=list, init=False)
    deleting_constraints: list[LinearConstraint] = field(default_factory=list, init=False)
    objective_functions: list[ObjectiveFunction] = field(default_factory=list, init=False)

    def __post_init__(self, solver_api: Type["AbstractSolverApi"]):
        self.set_solver(solver_api)

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

    def set_option(self, name: str, value) -> "Problem":
        "Sets the solver option"
        self.options[name] = value
        return self

    def set_options(self, options: dict[str, Any]) -> "Problem":
        "Sets a some solver options"
        for key, val in options.items():
            self.set_option(key, val)
        return self

    def add_var(self, variable: Variable) -> "Problem":
        "Adds a new column to the optimization model. Commit new variable with `problem.update()`."
        self.pending_variables += [variable]
        with suppress(ValueError):
            self.deleting_variables.remove(variable)
        return self

    def add_vars(self, *variables: list | dict | Variable) -> "Problem":
        "Adds some columns to the optimization model. Commit new variables with `problem.update()`."
        for list_of_variables in variables:
            if isinstance(list_of_variables, dict):
                list_of_variables = list_of_variables.values()
            elif isinstance(list_of_variables, Variable):
                list_of_variables = [list_of_variables]
            self.pending_variables += list(list_of_variables)
            for var in list_of_variables:
                with suppress(ValueError):
                    self.deleting_variables.remove(var)
        return self

    def del_var(self, variable: Variable | int) -> "Problem":
        """
        Marks a variable (object ou column index) to be deleted from the model.
        Commit deletion with `problem.update()`.
        """
        if isinstance(variable, int):
            variable_filter = [
                var
                for var in self.variables
                if var.column == variable
            ]
            if not variable_filter:
                raise Exception(f"No variable with column index {variable} was added to the model.")

            self.deleting_variables += [variable_filter[0]]
            return self

        if isinstance(variable, Variable):
            if variable in self.variables:
                self.deleting_variables += [variable]
                with suppress(ValueError):
                    self.pending_variables.remove(variable)
                return self

        raise TypeError("The decision variable should be of type `Variable` or a column index of type `int`.")

    def del_vars(self, *variables) -> "Problem":
        """
        Marks a set of variables (objects ou column indexes) to be deleted from the model.
        Commit deletion with `problem.update()`.
        """
        columns_in_model = [var.column for var in self.variables]
        for list_of_vars in variables:
            if isinstance(list_of_vars, dict):
                list_of_vars = list_of_vars.values()
            elif isinstance(list_of_vars, Variable):
                list_of_vars = [list_of_vars]
            for var in list_of_vars:
                if var.column in columns_in_model:
                    self.deleting_variables += [var]
                with suppress(ValueError):
                    self.pending_variables.remove(var)
        return self

    def add_constr(self, constr: LinearConstraint) -> "Problem":
        "Adds a new constraint to the model. Commit new constraint with `problem.update()`."
        self.pending_constraints += [constr]
        with suppress(ValueError):
            self.deleting_constraints.remove(constr)
        return self

    def add_constrs(self, *constrs: list | dict | LinearConstraint) -> "Problem":
        "Adds some constraints to the model. Commit new constraints with `problem.update()`."
        for list_of_constrs in constrs:
            if isinstance(list_of_constrs, dict):
                list_of_constrs = list_of_constrs.values()
            elif isinstance(list_of_constrs, LinearConstraint):
                list_of_constrs = [list_of_constrs]
            self.pending_constraints += list(list_of_constrs)
            for constr in list_of_constrs:
                with suppress(ValueError):
                    self.deleting_constraints.remove(constr)
        return self

    def del_constr(self, constr: LinearConstraint | int) -> "Problem":
        "Marks a constraint to be removed from the model. Commit deletion with `problem.update()`"
        if isinstance(constr, int):
            constr_filter = [
                constr_
                for constr_ in self.constraints
                if constr_.row == constr
            ]
            if not constr_filter:
                raise Exception(f"No constraint with row index {constr} was added to the model.")

            self.deleting_constraints += [constr_filter[0]]
            return self

        if isinstance(constr, LinearConstraint):
            if constr in self.constraints:
                self.deleting_constraints += [constr]
                with suppress(ValueError):
                    self.pending_constraints.remove(constr)
                return self

        raise TypeError("The constraint should be of type `LinearConstraint` or a row index of type `int`.")

    def del_constrs(self, *constraints: list | dict | LinearConstraint) -> "Problem":
        "Marks some constraints to be removed from the model. Commit deletion with `problem.update()`"
        rows_in_model = [constr.row for constr in self.constraints]
        for list_of_constrs in constraints:
            if isinstance(list_of_constrs, dict):
                list_of_constrs = list_of_constrs.values()
            elif isinstance(list_of_constrs, LinearConstraint):
                list_of_constrs = [list_of_constrs]

            for constr in list_of_constrs:
                if constr.row in rows_in_model:
                    self.deleting_constraints += [constr]
                with suppress(ValueError):
                    self.pending_constraints.remove(constr)
        return self

    def set_objective(self, objective: ObjectiveFunction | Variable | LinearExpression | float | int) -> "Problem":
        "Sets the objetive function to solve for."
        self.objective_functions = list()
        return self.add_objective(objective)

    def add_objectives(self, *objectives: ObjectiveFunction | Variable | LinearExpression | float | int) -> "Problem":
        "Adds some objective functions to solve for."
        for objective in objectives:
            self.add_objective(objective)
        return self

    def add_objective(self, objective: ObjectiveFunction | Variable | LinearExpression | float | int) -> "Problem":
        "Adds a new objective function to solve for."
        if isinstance(objective, (Variable, float, int)):
            objective += LinearExpression()
        if isinstance(objective, LinearExpression):
            objective = ObjectiveFunction(expression=objective)

        self.objective_functions += [objective]
        return self

    def set_solver(self, solver_api: Optional[Type["AbstractSolverApi"]] = None) -> "Problem":
        "Sets the solver from the given interface."
        self.solver = solver_api() if solver_api is not None else HighsApi()
        return self

    def update_del_vars(self):
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

    def update_del_constrs(self):
        for deleting_constr in self.deleting_constraints:
            self.solver.del_constr(deleting_constr)  # Delete row
            for idx in range(deleting_constr.row + 1, len(self.constraints)):
                self.constraints[idx].row -= 1
            self.constraints.remove(deleting_constr)
        self.deleting_constraints = []

    def update_add_vars(self):
        for idx, variable in enumerate(self.pending_variables):
            variable.column = idx + len(self.variables)
            variable.set_default_name(variable.column)
        self.solver.add_vars(self.pending_variables)
        self.variables += self.pending_variables
        self.pending_variables = []

    def update_add_constrs(self):
        for idx, constr in enumerate(self.pending_constraints):
            constr.row = idx + len(self.constraints)
            constr.set_default_name(constr.row)
        self.solver.add_constrs(self.pending_constraints)
        self.constraints += self.pending_constraints
        self.pending_constraints = []

    def update_vars(self):
        self.solver.update_vars(self.variables)

    def update(self) -> "Problem":
        """
        Deletes and adds any pending variables and constraints to the model,
        and sets the objective function.
        """
        self.update_del_vars()
        self.update_del_constrs()
        self.update_add_vars()
        self.update_add_constrs()
        self.update_vars()
        return self

    def set_hotstart(self) -> "Problem":
        columns, values = zip(*[
            (var.column, var.hotstart)
            for var in self.variables
            if var.hotstart is not None
            and var.column is not None
            and (
                (0 if var.vartype == VarType.BIN else (var.lowerbound if var.lowerbound is not None else -INF))
                <= var.hotstart
                <= (1 if var.vartype == VarType.BIN else (var.upperbound if var.upperbound is not None else INF))
            )
        ])
        self.solver.set_hotstart(columns, values)
        return self

    def _solve_preamble(self, update: bool = True, with_hotstart: bool = False) -> None:
        if self.solver is None:
            raise Exception("The solver api should be set before solving.")

        if update:
            self.update()

        if not self.variables:
            raise Exception("No variable was added to the problem.")

        if with_hotstart:
            self.set_hotstart()

    def set_solve_objective(self, objective: ObjectiveFunction):
        "Sets the current objective function to be solved for into the solver."
        self.current_objective = objective
        self.solver.set_objective(objective)

    def solve(self, update: bool = True, with_hotstart: bool = False, with_target: bool = False) -> "Problem":
        """
        Runs the solver for the optimization problem. This is a wrapper method for all optimization strategies:

            - Single objective: If only 1 objective was added to the problem, then solves for it.

            - Multiple objectives: If more than 1 objective was added to the problem, then solves for each one
            using the previous result value as a constraint to the next iteration.

            - Variables with target values: Adds a first step where the target values for the
            decision variables will try to be met. Any variable with target met will have its bounds
            changed to the target value, and then the optimization problem will be returned to its original
            form to be solved as usual.

        Args:

            - `update` (bool, default: True): Updates or not the actual solver model before solve.

            - `with_hotstart` (bool, default: False): Solves the problem using the hotstart solution.

            - `with_target` (bool, default: False): Whether or not it should solve for the decision
            variables targets first.
        """
        if with_target:
            return self.solve_withtarget(update, with_hotstart)

        if len(self.objective_functions) == 1:
            return self.solve_singleobjective(update, with_hotstart)

        return self.solve_multiobjective(update, with_hotstart)

    def solve_withtarget(self, update: bool = True, with_hotstart: bool = False) -> "Problem":
        self._solve_preamble(update, with_hotstart)

        # Filter variables which have a target value.
        variables_with_target = [var for var in self.variables if var.target is not None]

        # Add new auxiliary variables and constraints to compute the absolute value for the
        # difference between the solution and target.
        auxiliary_variables: list[Variable] = []
        for idx, var in enumerate(variables_with_target):
            aux_var = Variable(vartype=VarType.CNT, lowerbound=0)
            aux_var.column = idx + len(self.variables)
            aux_var.set_default_name(aux_var.column)
            auxiliary_variables += [aux_var]
        self.solver.add_vars(auxiliary_variables)

        auxiliary_constrs: list[LinearConstraint] = []
        for var, aux_var in zip(variables_with_target, auxiliary_variables):
            auxiliary_constrs += [(aux_var >= var - var.target)]
            auxiliary_constrs += [(aux_var >= -(var - var.target))]
        self.solver.add_constrs(auxiliary_constrs)

        # Add the objective function to minimize the sum of auxiliary variables.
        self.set_solve_objective(ObjectiveFunction(
            sum(aux_var/var.target for var, aux_var in zip(variables_with_target, auxiliary_variables)),
            is_minimization=True,
            name="target_var_values_objective"
        ))
        self.solver.run(self.options or dict())
        self.fetch_solution()

        # Back the original bounds up, and change them to the target value where applicable.
        original_bounds: dict[int, tuple[float, float]] = dict()
        for idx, (var, aux_var) in enumerate(zip(variables_with_target, auxiliary_variables)):
            if var.target is None and abs(var.target-var.value) < self.decimal_tol:
                original_bounds[idx] = (var.lowerbound, var.upperbound)
                var.lowerbound, var.upperbound = [var.target]*2
        self.update_vars()

        # Back the original hotstart values up, and change them to the solution found in the previous step.
        original_hotstart: list[float] = [None]*len(self.variables)
        for idx, var in enumerate(self.variables):
            original_hotstart[idx] = var.hotstart
            var.hotstart = var.value
        self.set_hotstart()

        # Delete auxiliary variables and constraints
        self.solver.del_vars(auxiliary_variables)
        self.solver.del_constrs(auxiliary_constrs)

        # Solve the original problem.
        self.solve(update, with_hotstart, False).fetch_solution()

        # Return bounds back to the original values.
        for var_idx, (lb, ub) in original_bounds.items():
            variables_with_target[var_idx].lowerbound = lb
            variables_with_target[var_idx].upperbound = ub
        self.update_vars()

        # Return hotstart back to the original values.
        for idx, hotstart in enumerate(original_hotstart):
            self.variables[idx].hotstart = hotstart

        return self

    def solve_singleobjective(self, update: bool = True, with_hotstart: bool = False) -> "Problem":
        "Runs the solver for the optimization problem with a single objective functions."
        self._solve_preamble(update, with_hotstart)
        objective = self.objective_functions[0]
        self.set_solve_objective(objective)
        options = objective.options if len(objective.options) > 0 else (self.options or dict())
        self.solver.run(options)
        return self.fetch_solution()

    def solve_multiobjective(self, update: bool = True, with_hotstart: bool = False) -> "Problem":
        """
        Runs the solver in a multiobjective fashion. It will solve for each objective setting the previous
        objective value as a constraint to the next iteration.
        """
        self._solve_preamble(update, with_hotstart)
        added_constrs = []
        for idx, objective in enumerate(self.objective_functions):
            if self.solver.show_log:
                if idx > 0:
                    print()
                obj_name = f"'{objective.name}' " if objective.name is not None else ""
                print(
                    f">> Solving for objective {obj_name}({idx+1} of {len(self.objective_functions)}, "
                    f"sense: {'Minimization' if objective.is_minimization else 'Maximization'})"
                )

            self.set_solve_objective(objective)

            options = objective.options if len(objective.options) > 0 else (self.options or dict())
            self.solver.run(options)
            self.fetch_solve_status()

            if (
                self.solve_status in [SolveStatus.FEASIBLE, SolveStatus.OPTIMUM]
                and idx < len(self.objective_functions) - 1
            ):
                self.solver.fetch_solution()
                self.solver.set_hotstart(list(range(len(self.solver.solution))), self.solver.solution)

                new_constr = (
                    objective.expression <= self.get_objectivefunction_value()
                    if objective.is_minimization
                    else objective.expression >= self.get_objectivefunction_value()
                )
                self.add_constr(new_constr)
                self.update()
                added_constrs += [new_constr]
            else:
                break

        return self.fetch_solution().del_constrs(added_constrs).update()

    def fetch_solve_status(self) -> "Problem":
        self.solver.fetch_solve_status()
        return self

    def fetch_solution(self) -> "Problem":
        "Retrieve all variable and dual values after a solve."
        self.solver.fetch_solve_status()
        if self.solve_status in [SolveStatus.FEASIBLE, SolveStatus.OPTIMUM]:
            self.solver.fetch_solution()
            for variable in self.variables:
                variable.value = self.solver.get_solution(variable)
            for constr in self.constraints:
                constr.dual = self.solver.get_dual(constr)
            self.current_objective.value = self.get_objectivefunction_value()
        return self

    def get_objectivefunction_value(self) -> float:
        return self.solver.get_objective_value()

    def get_solution(self, variable: Variable) -> float:
        "Returns the solution value of a variable."
        return self.solver.get_solution(variable)

    def get_dual(self, constraint: LinearConstraint) -> float:
        "Returns the dual value of a constraint."
        return self.solver.get_dual(constraint)

    def clear(self) -> "Problem":
        "Returns a fresh instance of the optimization problem and clears the solver object."
        if self.solver is not None:
            self.solver.clear()
        return Problem(name=self.name, solver_api=self.solver.__class__, options=(self.options or dict()))

    def clear_solver(self) -> "Problem":
        """
        Returns the optimization problem to a pre-update state and clears the solver object.
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

        self.current_objective = None

        return self

    def to_mps(self, path: str) -> None:
        """
        Exports the model to an MPS file. It will ignore pending variables and constraints,
        and will write only the last objective function added to the problem.
        """
        solver = HighsApi()
        solver.add_vars(self.variables)
        solver.add_constrs(self.constraints)
        solver.set_objective(self.objective_functions[-1])
        solver.to_mps(path)

    @staticmethod
    def load_mps(path: str,
                 name: Optional[str] = None,
                 solver_api: Optional["AbstractSolverApi"] = None,
                 options: Optional[dict[str, Any]] = None) -> "Problem":
        "Creates a new problem from a MPS file. It does not handle multiple objectives."
        variables, constraints, objective_function = HighsApi.load_mps(path)
        return (
            Problem(name=name, solver_api=solver_api, options=(options or dict()))
            .add_vars(variables)
            .add_constrs(constraints)
            .set_objective(objective_function)
            .update()
        )

    def rebuild_solver_model(self) -> "Problem":
        "Resets the solver model and rebuilds it according to the current optimization problem."
        self.solver.clear()
        self.solver.add_vars(self.variables)
        self.solver.add_constrs(self.constraints)

    def pull_from_solver(self) -> "Problem":
        """
        Pulls all variables, constraints, and the objective function from the actual model and returns a fresh
        instance of the optimization problem.
        """
        variables, constraints, objective_function = self.solver.pull_from_model()
        return (
            Problem(name=self.name, solver_api=self.solver.__class__, options=(self.options or dict()))
            .add_vars(variables)
            .add_constrs(constraints)
            .set_objective(objective_function)
            .update()
        )
