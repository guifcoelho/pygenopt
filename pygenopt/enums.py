from enum import Enum


class ConstraintSign(Enum):
    "The available constraint signs."
    LEQ = '<='
    GEQ = '>='
    EQ = '=='

class VarType(Enum):
    "Enumeration of decision varible types."
    BIN = 'bin'
    INT = 'int'
    CNT = 'cnt'

class SolveStatus(Enum):
    "Enumeration of the solve status."
    OPTIMUM = 'optimum'
    FEASIBLE = 'feasible'
    UNBOUNDED = 'unbounded'
    INFEASIBLE = 'infeasible'
    NOT_SOLVED = 'not_solved'
    UNKNOWN = 'unknown'