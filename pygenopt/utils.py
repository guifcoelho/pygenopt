from pygenopt import LinearExpression


def isiterable(obj):
    try:
        obj = iter(obj)
    except TypeError:
        return False
    else:
        return True

def isnumeric(obj):
    return isinstance(obj, (float, int))

_pysum = sum
def Sum(values):
    if isinstance(values, dict):
        values = list(values.values())
    return LinearExpression() + _pysum(values)

def Dot(values1, values2):
    if (
        not isiterable(values1) and not isnumeric(values1)
        and isiterable(values2) and not isnumeric(values2)
    ):
        raise TypeError("Arguments must be numeric or iterables.")

    if isinstance(values1, (int, float)) and isinstance(values2, (int, float)):
        return values1 * values2

    if isiterable(values1) and not isiterable(values2):
        return Dot(values1, [values2]*len(values1))

    if not isiterable(values1) and isiterable(values2):
        return Dot([values1]*len(values2), values2)

    return Sum(va1 * va2 for va1, va2 in zip(values1, values2))
