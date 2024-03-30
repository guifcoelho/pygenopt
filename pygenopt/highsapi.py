from dataclasses import dataclass
from typing import Any, Optional

import highspy

from pygenopt import (
    LinearExpression, Variable, LinearConstraint, ConstraintSign, SolverApi, VarType, SolveStatus
)


@dataclass
class HiGHS(SolverApi):

    solver_name = 'HiGHS'

    @property
    def show_log(self):
        if self.model is None:
            raise ValueError("The solver model was not set.")
        return bool(self.get_option('output_flag'))

    def init_model(self):
        self.model = highspy.Highs()
        self._set_log(False)
        return self

    def _set_log(self, flag: bool):
        self.set_option('output_flag', 'true' if flag else 'false')
        return self

    def get_version(self):
        return f"v{self.model.version()}"

    def _set_integrality(self, variables: list[Variable]):
        integer_variables = [
            variable
            for variable in variables
            if variable.vartype in [VarType.BIN, VarType.INT]
        ]
        if len(integer_variables) > 0:
            self.model.changeColsIntegrality(
                len(integer_variables),
                [variable.column for variable in integer_variables],
                [highspy.HighsVarType.kInteger] * len(integer_variables)
            )

    def add_var(self, variable: Variable):
        lb = [0 if variable.vartype == VarType.BIN else (variable.lowerbound or -highspy.kHighsInf)]
        ub = [1 if variable.vartype == VarType.BIN else (variable.lowerbound or highspy.kHighsInf)]
        self.model.addVars(1, lb, ub)
        self._set_integrality([variable])
        self.model.passColName(variable.column, variable.default_name)
        return self

    def add_vars(self, variables: list[Variable]):
        lbs = [
            0
            if var.vartype == VarType.BIN
            else (var.lowerbound or -highspy.kHighsInf)
            for var in variables
        ]
        ubs = [
            1
            if var.vartype == VarType.BIN
            else (var.lowerbound or highspy.kHighsInf)
            for var in variables
        ]
        self.model.addVars(len(variables), lbs, ubs)
        self._set_integrality(variables)
        for var in variables:
            self.model.passColName(var.column, var.default_name)
        return self

    def del_var(self, variable: Variable):
        self.model.deleteCols(1, [variable.column])
        return self

    def del_vars(self, variables: list[Variable]):
        self.model.deleteCols(len(variables), [variable.column for variable in variables])
        return self

    def add_constr(self, constr: LinearConstraint):
        vars, coefs = zip(*list(constr.expression.elements.items()))
        self.model.addRow(
            -highspy.kHighsInf if constr.sign == ConstraintSign.LEQ else -constr.expression.constant,
            highspy.kHighsInf if constr.sign == ConstraintSign.GEQ else -constr.expression.constant,
            len(vars),
            [var.column for var in vars],
            coefs
        )
        return self

    def set_objective(self, objetive_function: LinearExpression, is_minimization: bool = True):
        vars, coefs = zip(*list(objetive_function.elements.items()))
        self.model.changeColsCost(len(vars), [var.column for var in vars], coefs)
        self.model.changeObjectiveOffset(objetive_function.constant)
        self.model.changeObjectiveSense(
            highspy.ObjSense.kMinimize if is_minimization else highspy.ObjSense.kMaximize
        )
        return self

    def set_option(self, name: str, value: Any):
        self.model.setOptionValue(name, value)
        return self

    def get_option(self, name: str):
        return self.model.getOptionValue(name)

    def fetch_solution(self):
        self.solution = list(self.model.getSolution().col_value)
        return self

    def fetch_duals(self) -> SolverApi:
        self.duals = list(self.model.getSolution().row_dual)
        return self

    def get_solution(self, variable: Variable) -> float | None:
        if self.solution is None:
            self.fetch_solution()
        return self.solution[variable.column]

    def get_dual(self, constraint: LinearConstraint) -> float | None:
        if self.duals is None:
            self.fetch_duals()
        return self.duals[constraint.row]

    def fetch_solve_status(self):
        match self.model.getModelStatus():
            case highspy.HighsModelStatus.kOptimal:
                self.solve_status = SolveStatus.OPTIMUM
            case highspy.HighsModelStatus.kInfeasible:
                self.solve_status = SolveStatus.INFEASIBLE
            case highspy.HighsModelStatus.kUnbounded:
                self.solve_status = SolveStatus.UNBOUNDED
            case _:
                self.solve_status = SolveStatus.UNKNOWN

    def set_hotstart(self, variables: list[Variable]):
        # sol = highspy.HighsSolution()
        # sol.col_value = [var.value or 0 for var in variables]
        # self.model.setSolution(sol)

        current_show_log = self.show_log

        columns, values, lbs, ubs = zip(*[
            (var.column, var.value, var.lowerbound, var.upperbound)
            for var in variables
            if var.value is not None and var.column is not None
        ])

        self.model.changeColsBounds(len(columns), columns, values, values)
        self.set_option('mip_rel_gap', highspy.kHighsInf)
        self._set_log(False)

        self.model.run()
        self.fetch_solve_status()

        self.set_option('mip_rel_gap', 0)
        self._set_log(current_show_log)
        self.model.changeColsBounds(len(columns), columns, lbs, ubs)

        if self.solve_status in [SolveStatus.OPTIMUM, SolveStatus.FEASIBLE]:
            sol = highspy.HighsSolution()
            sol.col_value = self.model.getSolution().col_value
            self.model.setSolution(sol)

    def run(self,
            options: Optional[dict[str, Any]] = None,
            hotstart: Optional[list[Variable]] = None):

        self.set_options(options or dict())

        if self.show_log:
            print(f"Solver: {self.solver_name} {self.get_version()}")

        if hotstart is not None:
            self.set_hotstart(hotstart)

        self.model.run()

        self.fetch_solve_status()
        return self

    def clear(self):
        super().clear()
        self.init_model()
        return self
