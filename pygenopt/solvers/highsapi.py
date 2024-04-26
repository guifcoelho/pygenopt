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
        if not variables:
            return self

        lbs, ubs = zip(*[
            (
                -highspy.kHighsInf if var.lowerbound is None else var.lowerbound,
                highspy.kHighsInf if var.upperbound is None else var.upperbound
            )
            for var in variables
        ])
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
        if not variables:
            return self

        self.model.deleteCols(len(variables), [variable.column for variable in variables])
        return self

    def update_var(self, variable: Variable) -> "HighsApi":
        return self.update_vars([variable])

    def update_vars(self, variables: list[Variable]) -> "HighsApi":
        if not variables:
            return self

        lbs, ubs, columns = zip(*[
            (
                -highspy.kHighsInf if var.lowerbound is None else var.lowerbound,
                highspy.kHighsInf if var.upperbound is None else var.upperbound,
                var.column
            )
            for var in variables
        ])
        self.model.changeColsBounds(len(variables), columns, lbs, ubs)

        kInteger = highspy.HighsVarType.kInteger
        kContinuous = highspy.HighsVarType.kContinuous
        integralities = [
            kInteger if var.vartype in [VarType.BIN, VarType.INT] else kContinuous
            for var in variables
        ]
        self.model.changeColsIntegrality(len(variables), columns, integralities)

        for var in variables:
            self.model.passColName(var.column, var.default_name)

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
        if not constrs:
            return self
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

    def pull_from_model(self) -> tuple[list[Variable], list[LinearConstraint], ObjectiveFunction]:
        _, _, obj_coefs, lbs, ubs, _ = self.model.getCols(self.model.numVars, list(range(self.model.numVars)))
        integrality = [vartype == highspy.HighsVarType.kInteger for vartype in self.model.getLp().integrality_]
        if not integrality:
            integrality = [False] * len(obj_coefs)

        variables = []
        for column, (lb, ub, is_int) in enumerate(zip(lbs, ubs, integrality)):
            vartype = (
                VarType.BIN if lb == 0 and ub == 1 and is_int
                else (
                    VarType.INT if is_int else VarType.CNT
                )
            )
            variables += [
                Variable(
                    name=self.model.getColName(column)[1],
                    vartype=vartype,
                    lowerbound=None if vartype == VarType.BIN or lb == -highspy.kHighsInf else lb,
                    upperbound=None if vartype == VarType.BIN or ub == highspy.kHighsInf else ub,
                )
            ]

        objective_function = ObjectiveFunction(
            sum(var * coef for var, coef in zip(variables, obj_coefs))
            + self.model.getObjectiveOffset()[1],
            is_minimization=self.model.getObjectiveSense()[1] == highspy.ObjSense.kMinimize
        )

        constraints = dict()
        for column, var in enumerate(variables):
            _, rows, coefs = self.model.getColEntries(column)
            for row, coef in zip(rows, coefs):
                constraints[row] = constraints.get(row, LinearExpression()) + (var * coef)

        _, _, constr_lhs, constr_rhs, _ = self.model.getRows(self.model.getNumRow(), list(range(self.model.getNumRow())))
        for row, (constr_lhs, constr_rhs) in enumerate(zip(constr_lhs, constr_rhs)):
            if constr_lhs == -highspy.kHighsInf and constr_rhs != highspy.kHighsInf:
                constraints[row] = (constraints[row] <= constr_rhs)

            elif constr_lhs != -highspy.kHighsInf and constr_rhs == highspy.kHighsInf:
                constraints[row] = LinearConstraint(constraints[row] >= constr_lhs)

            elif constr_lhs != -highspy.kHighsInf and constr_rhs != highspy.kHighsInf and constr_lhs == constr_rhs:
                constraints[row] = LinearConstraint(constraints[row] == constr_lhs)
            else:
                raise ValueError(f"Cannot define constraint type from {constr_lhs} <= constr <= {constr_rhs}")

        _, list_of_constraints = zip(*sorted(list(constraints.items()), key=lambda el: el[0]))

        return variables, list_of_constraints, objective_function

    @staticmethod
    def load_mps(path: str) -> tuple[list[Variable], list[LinearConstraint], ObjectiveFunction]:
        """
        Loads a complete problem with highspy and pulls the variables,
        constraints and objective function from the model.
        It does not handle multiple objectives.
        """
        prob = HighsApi()._set_log(False)
        prob.model.readModel(path)
        return prob.pull_from_model()
