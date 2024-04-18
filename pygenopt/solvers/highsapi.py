from dataclasses import dataclass, field
from typing import Any, Optional

import highspy

from pygenopt import *
from pygenopt.enums import ConstraintSign
from pygenopt.solvers.abstractsolverapi import AbstractSolverApi


@dataclass
class HighsApi(AbstractSolverApi):

    solver_name = 'HiGHS'
    solution: list[float] | None = field(default=None, init=False, repr=False)
    duals: list[float] | None = field(default=None, init=False, repr=False)
    _hotstart_solution: list[float] | None = field(default=None, init=False, repr=False)

    @property
    def show_log(self) -> bool:
        if self.model is None:
            raise ValueError("The solver model was not set.")
        return bool(self.get_option('output_flag'))

    def init_model(self) -> "HighsApi":
        self.model = highspy.Highs()
        self._set_log(False)
        return self

    def _set_log(self, flag: bool) -> "HighsApi":
        self.set_option('output_flag', 'true' if flag else 'false')
        return self

    def get_version(self) -> str:
        return f"v{self.model.version()}"

    def add_var(self, variable: Variable) -> "HighsApi":
        return self.add_vars([variable])

    def add_vars(self, variables: list[Variable]) -> "HighsApi":
        lbs = [
            0 if var.vartype == VarType.BIN else (-highspy.kHighsInf if var.lowerbound is None else var.lowerbound)
            for var in variables
        ]
        ubs = [
            1 if var.vartype == VarType.BIN else (highspy.kHighsInf if var.upperbound is None else var.upperbound)
            for var in variables
        ]
        self.model.addVars(len(variables), lbs, ubs)
        for var in variables:
            self.model.passColName(var.column, var.default_name)
            if var.vartype in [VarType.BIN, VarType.INT]:
                self.model.changeColsIntegrality(1, [var.column], [highspy.HighsVarType.kInteger])

        return self

    def del_var(self, variable: Variable) -> "HighsApi":
        self.model.deleteCols(1, [variable.column])
        return self

    def del_vars(self, variables: list[Variable]) -> "HighsApi":
        self.model.deleteCols(len(variables), [variable.column for variable in variables])
        return self

    def add_constr(self, constr: LinearConstraint) -> "HighsApi":
        vars, coefs = zip(*list(constr.expression.elements.items()))
        for var in vars:
            if var.column is None:
                raise Exception("All variables need to be added to the model prior to adding constraints.")

        self.model.addRow(
            -highspy.kHighsInf if constr.sign == ConstraintSign.LEQ else constr.expression.constant,
            highspy.kHighsInf if constr.sign == ConstraintSign.GEQ else constr.expression.constant,
            len(vars),
            [var.column for var in vars],
            coefs
        )
        self.model.passRowName(constr.row, constr.default_name)
        return self

    def del_constr(self, constr: LinearConstraint) -> "HighsApi":
        self.model.deleteRows(1, [constr.row])
        return self

    def del_constrs(self, constrs: list[LinearConstraint]) -> "HighsApi":
        self.model.deleteRows(len(constrs), [constr.row for constr in constrs])
        return self

    def set_objective(self, objetive_function: ObjectiveFunction | Variable | LinearExpression | float | int) -> "HighsApi":
        if isinstance(objetive_function, (Variable | float | int)):
            objetive_function += LinearExpression()
        if isinstance(objetive_function, LinearExpression):
            objetive_function = ObjectiveFunction(expression=objetive_function)

        num_vars = self.model.numVars
        self.model.changeColsCost(num_vars, list(range(num_vars)), [0]*num_vars)
        if len(objetive_function.expression.elements) > 0:
            vars, coefs = zip(*list(objetive_function.expression.elements.items()))
            self.model.changeColsCost(len(vars), [var.column for var in vars], coefs)

        self.model.changeObjectiveOffset(objetive_function.expression.constant)
        self.model.changeObjectiveSense(
            highspy.ObjSense.kMinimize if objetive_function.is_minimization else highspy.ObjSense.kMaximize
        )
        return self

    def set_option(self, name: str, value: Any) -> "HighsApi":
        self.model.setOptionValue(name, value)
        return self

    def get_option(self, name: str) -> Any:
        return self.model.getOptionValue(name)

    def get_objective_value(self) -> float | None:
        return self.model.getObjectiveValue()

    def fetch_solution(self) -> "HighsApi":
        solver_solution = self.model.getSolution()
        self.solution = list(solver_solution.col_value)
        self.duals = list(solver_solution.row_dual)
        return self

    def get_solution(self, variable: Variable) -> float | None:
        if self.solution is None:
            self.fetch_solution()
        return self.solution[variable.column]

    def get_dual(self, constraint: LinearConstraint) -> float | None:
        if self.duals is None:
            self.fetch_duals()
        return self.duals[constraint.row]

    def fetch_solve_status(self) -> "HighsApi":
        match self.model.getModelStatus():
            case highspy.HighsModelStatus.kOptimal:
                self.solve_status = SolveStatus.OPTIMUM
            case highspy.HighsModelStatus.kInfeasible:
                self.solve_status = SolveStatus.INFEASIBLE
            case highspy.HighsModelStatus.kUnbounded:
                self.solve_status = SolveStatus.UNBOUNDED
            case _:
                self.solve_status = SolveStatus.UNKNOWN
        return self

    def set_hotstart(self, columns: list[int], values: list[float]) -> "HighsApi":
        # With HiGHS, the hotstart solution should be set just before a new execution.
        # When the model is changed the hotstart solution will then be reset.
        # Therefore, the hotstart solution will be kept in a list and added later into the model.

        # Also, the solver still lacks a clear way to add a partial solution into the model,
        # therefore we will fix the variables values to the hotstart solution, and then capture the
        # complete solution (if feasible) to add later into the model.
        # See https://github.com/ERGO-Code/HiGHS/discussions/1401.

        current_show_log = self.show_log
        self._hotstart_solution = None
        num_vars = self.model.numVars

        if num_vars == len(columns):
            _, sorted_values_by_index = zip(*sorted(
                [(idx, val) for idx, val in enumerate(values)],
                key=lambda el: el[0]
            ))
            self._hotstart_solution = list(sorted_values_by_index)
            return self

        _, _, costs, lbs, ubs, *_ = self.model.getCols(num_vars, list(range(num_vars)))
        self.set_objective(0)
        self.model.changeColsBounds(len(columns), columns, values, values)
        self.set_option('mip_rel_gap', highspy.kHighsInf)
        self._set_log(False)

        self.model.run()
        self.fetch_solve_status()

        self.set_option('mip_rel_gap', 0)
        self._set_log(current_show_log)
        self.model.changeColsBounds(num_vars, list(range(num_vars)), lbs, ubs)
        self.model.changeColsCost(num_vars, list(range(num_vars)), costs)

        if self.solve_status in [SolveStatus.OPTIMUM, SolveStatus.FEASIBLE]:
            self._hotstart_solution = self.model.getSolution().col_value

        return self

    def run(self, options: Optional[dict[str, Any]] = None) -> "HighsApi":
        self._set_log(True)
        self.set_options(options or dict())

        if self.show_log:
            print(f"Solver: {self.solver_name} {self.get_version()}")

        if self._hotstart_solution is not None:
            sol = highspy.HighsSolution()
            sol.col_value = self._hotstart_solution
            self.model.setSolution(sol)

        self.model.run()

        return self

    def to_mps(self, path: str) -> None:
        self.model.writeModel(path)
