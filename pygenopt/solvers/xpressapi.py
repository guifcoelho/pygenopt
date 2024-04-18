from dataclasses import dataclass
from typing import Any, Optional

import xpress as xp

from pygenopt import *
from pygenopt.enums import ConstraintSign
from pygenopt.solvers.abstractsolverapi import AbstractSolverApi


@dataclass
class XpressApi(AbstractSolverApi):
    solver_name = 'Xpress'

    def __post_init__(self):
        self.init_model()

    @property
    def show_log(self):
        return self.model.getControl('OUTPUTLOG') > 0

    def init_model(self) -> "XpressApi":
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

        return solver_var

    def add_var(self, variable: Variable) -> "XpressApi":
        self.model.addVariable(self._to_xpvar(variable))
        return self

    def add_vars(self, variables: list[Variable]) -> "XpressApi":
        self.model.addVariable(*[self._to_xpvar(var) for var in variables])
        return self

    def del_var(self, variable: Variable) -> "XpressApi":
        self.model.delVariable(variable.column)
        return self

    def del_vars(self, variables: list[Variable]) -> "XpressApi":
        self.model.delVariable([variable.column for variable in variables])
        return self

    def add_constr(self, constraint: LinearConstraint) -> "XpressApi":
        for var in constraint.expression.elements:
            if var.column is None:
                raise Exception("All variables need to be added to the model prior to adding constraints.")

        xpvars, coefs = [], []
        if len(constraint.expression.elements) > 0:
            vars, coefs = zip(*list(constraint.expression.elements.items()))
            xpvars = self.model.getVariable([var.column for var in vars])
        lhs = xp.Sum([xpvar * coef for xpvar, coef in zip(xpvars, coefs)])

        match constraint.sign:
            case ConstraintSign.EQ:
                self.model.addConstraint(lhs == constraint.expression.constant)
            case ConstraintSign.LEQ:
                self.model.addConstraint(lhs <= constraint.expression.constant)
            case ConstraintSign.GEQ:
                self.model.addConstraint(lhs >= constraint.expression.constant)

        return self

    def del_constr(self, constraint: LinearConstraint) -> "XpressApi":
        self.model.delConstraint(constraint.row)
        return self

    def del_constrs(self, constraints: list[LinearConstraint]) -> "XpressApi":
        self.model.delConstraint([constraint.row for constraint in constraints])
        return self

    def set_objective(self, objetive_function: ObjectiveFunction | Variable | LinearExpression | float | int) -> "XpressApi":
        if isinstance(objetive_function, (Variable | float | int)):
            objetive_function += LinearExpression()
        if isinstance(objetive_function, LinearExpression):
            objetive_function = ObjectiveFunction(expression=objetive_function)

        xpvars, coefs = [], []
        if len(objetive_function.expression.elements) > 0:
            vars, coefs = zip(*list(objetive_function.expression.elements.items()))
            xpvars = self.model.getVariable([var.column for var in vars])

        self.model.setObjective(
            xp.Sum([xpvar * coef for xpvar, coef in zip(xpvars, coefs)]) + objetive_function.expression.constant,
            sense=xp.minimize if objetive_function.is_minimization else xp.maximize
        )
        return self

    def set_option(self, name: str, value) -> "XpressApi":
        self.model.setControl(name, value)
        return self

    def get_option(self, name: str) -> Any:
        return self.model.getControl(name)

    def fetch_solution(self) -> "XpressApi":
        self.solution = list(self.model.getSolution())
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

    def fetch_solve_status(self) -> "XpressApi":
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

    def set_hotstart(self, columns: list[int], values: list[float]) -> "XpressApi":
        if len(columns) == self.model.attributes.cols:
            _, sorted_values_by_index = zip(*sorted(
                [(idx, val) for idx, val in enumerate(values)],
                key=lambda el: el[0]
            ))
            self.model.loadmipsol(list(sorted_values_by_index))
        else:
            self.model.addmipsol(values, columns, "hotstart")
        return self

    def run(self, options: Optional[dict[str, Any]] = None) -> "XpressApi":
        self.set_options(options)
        self.model.optimize()
        return self

    def to_mps(self, path: str) -> None:
        self.model.write(path, 'x')
