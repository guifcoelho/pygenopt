from pygenopt import LinearExpression


_pysum = sum
def sum(values):
    if isinstance(values, dict):
        values = list(values.values())
    return LinearExpression() + _pysum(values)
