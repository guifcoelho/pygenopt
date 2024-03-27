from typing import Any, Optional

import highspy

from pygenopt import LinearExpression, Variable, LinearConstraint, ConstraintSign
from pygenopt.abstractsolverapi import AbstractSolverApi


class HiGHS(AbstractSolverApi):

    def __init__(self, name: str = "NoName", options: Optional[dict[str, Any]] = None):
        super().__init__(name=name, options=options or dict())
        self.model = highspy.Highs()
        self.model.setOptionValue('output_flag', False)

    def addvar(self, variable: Variable):
        super().addvar(variable)

        self.model.addVars(
            1,
            [variable.lowerbound or -highspy.kHighsInf],
            [variable.upperbound or highspy.kHighsInf]
        )

    def addvars(self, *variables):
        super().addvars(variables)

        if len(variables) == 1 and isinstance(variables[0], (list, tuple)):
            variables = variables[0]

        self.model.addVars(
            len(variables),
            [var.lowerbound or -highspy.kHighsInf for var in variables],
            [var.upperbound or highspy.kHighsInf for var in variables]
        )
        return self

    def _addrow(self, constr: LinearConstraint):
        vars, coefs = zip(*list(constr.expression.elements.items()))
        self.model.addRow(
            -highspy.kHighsInf if constr.sign == ConstraintSign.LEQ else -constr.expression.constant,
            highspy.kHighsInf if constr.sign == ConstraintSign.GEQ else -constr.expression.constant,
            len(vars),
            [self.getcolidx(var) for var in vars],
            coefs
        )

    def addconstr(self, constr: LinearConstraint):
        super().addconstr(constr)
        self._addrow(constr)
        return self

    def addconstrs(self, *constrs):
        super().addconstrs(constrs)

        if len(constrs) == 1 and isinstance(constrs[0], (list, tuple)):
            constrs = constrs[0]

        for constr in constrs:
            self._addrow(constr)
        return self

    def setobj(self, objetive_function: LinearExpression | Variable | float, minimization=True):
        super().setobj(objetive_function, minimization)

        vars, coefs = zip(*list(self.obj.elements.items()))
        self.model.changeColsCost(len(vars), [self.getcolidx(var) for var in vars], coefs)
        self.model.changeObjectiveSense(
            highspy.ObjSense.kMinimize if self.minimization else highspy.ObjSense.kMaximize
        )
        return self

    def run(self) -> 'AbstractSolverApi':
        self.model.setOptionValue('output_flag', True)

        if self.options is not None:
            for key, val in self.options.items():
                self.model.setOptionValue(key, val)

        self.model.run()

        self.solution = list(self.model.getSolution().col_value)
        return self

    def getval(self, variable: Variable):
        return self.solution[self.getcolidx(variable)]
