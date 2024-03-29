from dataclasses import dataclass
from typing import Any, Optional

import highspy

from pygenopt import (
    LinearExpression, Variable, LinearConstraint, ConstraintSign, SolverApi, VarType
)


@dataclass
class HiGHS(SolverApi):

    solver_name = 'HiGHS'

    def init_model(self):
        self.model = highspy.Highs()
        self.set_option('output_flag', False)

    def _set_integer_variables(self, variables: list[Variable]):
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
        self.model.addVars(
            1,
            [variable.lowerbound or -highspy.kHighsInf],
            [variable.upperbound or highspy.kHighsInf]
        )
        self._set_integer_variables([variable])
        return self

    def add_vars(self, variables: list[Variable]):
        self.model.addVars(
            len(variables),
            [var.lowerbound or -highspy.kHighsInf for var in variables],
            [var.upperbound or highspy.kHighsInf for var in variables]
        )
        self._set_integer_variables(variables)
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

    def add_constrs(self, constrs: list[LinearConstraint]):
        for constr in constrs:
            self.add_constr(constr)
        return self

    def set_objective(self, objetive_function: LinearExpression, is_minimization: bool = True):
        vars, coefs = zip(*list(objetive_function.elements.items()))
        self.model.changeColsCost(len(vars), [var.column for var in vars], coefs)
        self.model.changeObjectiveSense(
            highspy.ObjSense.kMinimize if is_minimization else highspy.ObjSense.kMaximize
        )
        return self

    def set_option(self, name: str, value: Any):
        self.model.setOptionValue(name, value)
        return self

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

    def run(self, options: Optional[dict[str, Any]] = None):
        self.set_option('output_flag', True)

        for key, val in (options or dict()).items():
            self.model.setOptionValue(key, val)

        self.model.run()
        return self
