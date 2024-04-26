from pygenopt import LinearExpression


_pysum = sum
def Sum(values):
    if isinstance(values, dict):
        values = list(values.values())
    return LinearExpression() + _pysum(values)

def Dot(coefs, variables):
    return Sum(coef * var for coef, var in zip(coefs, variables))
