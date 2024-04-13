from dataclasses import dataclass
from typing import Any, Optional

import xpress as xp

from pygenopt import (
    LinearExpression, Variable, LinearConstraint, ObjectiveFunction,
    ConstraintSign, SolverApi, VarType,
    SolveStatus
)


@dataclass
class Xpress(SolverApi):
    solver_name = 'Xpress'

    def __post_init__(self):
        self.init_model()

    def init_model(self) -> 'Xpress':
        self.model = xp.problem()
        return self

    def _to_xpvar(self, variable: Variable):
        lb = -xp.infinity if variable.lowerbound is None else variable.lowerbound
        ub = xp.infinity if variable.upperbound is None else variable.upperbound
        match variable.vartype:
            case VarType.BIN:
                solver_var = xp.var(name=variable.default_name, vartype=xp.binary)
            case VarType.INT:
                solver_var = xp.var(name=variable.default_name, vartype=xp.integer, lb=lb, ub=ub)
            case VarType.CNT:
                solver_var = xp.var(name=variable.default_name, vartype=xp.continuous, lb=lb, ub=ub)

        variable.set_solvervar(solver_var)
        return variable.solver_var

    def add_var(self, variable: Variable) -> 'Xpress':
        self.model.addVariable(self._to_xpvar(variable))
        return self

    def add_vars(self, variables: list[Variable]) -> 'Xpress':
        self.model.addVariable(*[self._to_xpvar(var) for var in variables])
        return self

    def del_var(self, variable: Variable) -> 'Xpress':
        raise NotImplementedError()

    def del_vars(self, variables: list[Variable]) -> 'Xpress':
        raise NotImplementedError()

    def add_constr(self, constraint: LinearConstraint) -> 'Xpress':
        for var in constraint.expression.elements:
            if var.column is None or var.solver_var is None:
                raise Exception("All variables need to be added to the model prior to adding constraints.")

        lhs = xp.Sum([var.solver_var * coef for var, coef in constraint.expression.elements.items()])
        match constraint.sign:
            case ConstraintSign.EQ:
                self.model.addConstraint(lhs == constraint.expression.constant)
            case ConstraintSign.LEQ:
                self.model.addConstraint(lhs <= constraint.expression.constant)
            case ConstraintSign.GEQ:
                self.model.addConstraint(lhs >= constraint.expression.constant)

        return self

    def set_objective(self, objetive_function: ObjectiveFunction | Variable | LinearExpression | float | int) -> "Xpress":
        if isinstance(objetive_function, (Variable | float | int)):
            objetive_function += LinearExpression()
        if isinstance(objetive_function, LinearExpression):
            objetive_function = ObjectiveFunction(expression=objetive_function)

        self.model.setObjective(
            xp.Sum([var.solver_var * coef for var, coef in objetive_function.expression.elements.items()]),
            sense=xp.minimize if objetive_function.is_minimization else xp.maximize
        )
        return self

    def set_option(self, name: str, value) -> 'Xpress':
        self.model.setControl(name, value)
        return self

    def get_option(self, name: str) -> Any:
        return self.model.getControl(name)

    def fetch_solution(self) -> 'Xpress':
        self.solution = list(self.model.getSolution())
        return self

    def fetch_duals(self) -> 'Xpress':
        self.duals = list(self.model.getDual())
        return self

    def get_objective_value(self) -> float:
        return self.model.getObjVal()

    def get_solution(self, variable: Variable) -> float:
        if self.solution is None:
            self.fetch_solution()
        return self.solution[variable.column]

    def get_dual(self, constraint: LinearConstraint) -> float:
        if self.duals is None:
            self.fetch_duals()
        return self.duals[constraint.row]

    def fetch_solve_status(self) -> 'Xpress':
        match self.model.getAttrib('SOLSTATUS'):
            case xp.SolStatus.OPTIMAL:
                self.solve_status = SolveStatus.OPTIMUM
            case xp.SolStatus.INFEASIBLE:
                self.solve_status = SolveStatus.INFEASIBLE
            case xp.SolStatus.UNBOUNDED:
                self.solve_status = SolveStatus.UNBOUNDED
            case xp.SolStatus.FEASIBLE:
                self.solve_status = SolveStatus.FEASIBLE
            case _:
                self.solve_status = SolveStatus.UNKNOWN

        return self

    def set_hotstart(self, columns: list[int], values: list[float]) -> 'Xpress':
        self.model.addmipsol(values, columns, "hotstart")
        return self

    def run(self, options: Optional[dict[str, Any]] = None) -> 'Xpress':
        self.set_options(options)
        self.model.optimize()
        return self

    def run_multiobjective(self, objectives: list[tuple[LinearExpression, bool, Optional[dict[str, Any]]]]) -> 'Xpress':
        raise NotImplementedError()

    def to_mps(self, path: str) -> None:
        self.model.write(path, 'x')
