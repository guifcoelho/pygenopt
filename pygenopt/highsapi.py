from dataclasses import dataclass, field
from typing import Any, Optional

import highspy

from pygenopt import (
    LinearExpression, Variable, LinearConstraint, ConstraintSign, SolverApi, VarType, SolveStatus
)


@dataclass
class HiGHS(SolverApi):

    solver_name = 'HiGHS'
    _hotstart_solution: list[float] | None = field(default=None, init=False, repr=False)

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

    def set_objective(self, objetive_function: LinearExpression | Variable | float, is_minimization: bool = True):
        objetive_function += LinearExpression()
        num_vars = self.model.numVars
        self.model.changeColsCost(num_vars, list(range(num_vars)), [0]*num_vars)
        if len(objetive_function.elements) > 0:
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

    def get_objective_value(self):
        return self.model.getObjectiveValue()

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

    def set_hotstart(self, columns: list[int], values: list[float]):
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
            self._hotstart_solution = values
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

    def run(self, options: Optional[dict[str, Any]] = None):
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

    def run_multiobjective(self, objectives: list[tuple[LinearExpression, bool, Optional[dict[str, Any]]]]):
        for idx, (objectivefunction, is_minimization, options) in enumerate(objectives):
            if self.show_log:
                print(f">> Solving for objective #{idx+1}")

            self.set_objective(objectivefunction, is_minimization)
            self.run(options or dict())
            self.fetch_solve_status()
            if self.solve_status in [SolveStatus.FEASIBLE, SolveStatus.OPTIMUM]:
                self.fetch_solution()
                self.set_hotstart(list(range(len(self.solution))), self.solution)
                self.add_constr(objectivefunction == self.get_objective_value())

        return self

    def clear(self):
        super().clear()
        self.init_model()
        return self
