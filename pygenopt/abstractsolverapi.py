"The abstract solver interface"

from typing import Any
from dataclasses import dataclass, field

from pygenopt import Variable, LinearConstraint, LinearExpression

@dataclass
class AbstractSolverApi():
    "The abstract solver API class"

    name: str = "NoName"
    options: dict[str, Any] = field(default_factory=dict)
    model: Any = field(default=None, init=False)
    variables: list[Variable] = field(default_factory=list, init=False)
    constraints: list[LinearConstraint] = field(default_factory=list, init=False)
    obj: LinearExpression = field(default=None, init=False)
    minimization: bool = field(default=True, init=False)
    solution: Any = field(default=None, init=False)

    def setoption(self, name: str, value):
        "Sets the solver option"
        self.options[name] = value
        return self

    def setoptions(self, options: dict[str, Any]):
        "Sets a some solver options"
        for key, val in options.items():
            self.setoption(key, val)
        return self

    def addvar(self, variable: Variable):
        "Adds a new column to the optimization model."
        self.variables += [variable]
        return self

    def addvars(self, *variables):
        "Adds some columns to the optimization model."
        if len(variables) == 1 and isinstance(variables[0], (list, tuple)):
            variables = variables[0]

        self.variables += variables
        return self

    def getcolidx(self, variable: Variable):
        "Returns the column index of the variable"
        for idx, var in enumerate(self.variables):
            if var is variable:
                return idx
        raise ValueError(f"Variable {variable} not in the list of variables.")

    def addconstr(self, constr: LinearConstraint):
        "Adds a new constraint to the model."
        self.constraints += [constr]
        return self

    def addconstrs(self, *constrs):
        "Adds some constraints to the model."
        if len(constrs) == 1 and isinstance(constrs[0], (list, tuple)):
            constrs = constrs[0]
        self.constraints += list(constrs)
        return self

    def run(self) -> 'AbstractSolverApi':
        "Runs the solver for the optimization problem."

    def getval(self, variable: Variable) -> float:
        "Returns the value of the decision variable after solving."

    def setobj(self, objetive_function: LinearExpression, minimization = True):
        "Sets the objetive function to solve for"
        self.obj = LinearExpression() + objetive_function
        self.minimization = minimization
        return self
