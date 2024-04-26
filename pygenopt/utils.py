from pygenopt import LinearExpression


_pysum = sum
def Sum(values):
    if isinstance(values, dict):
        values = list(values.values())
    return LinearExpression() + _pysum(values)

def Dot(values1, values2):
    def isiterator(obj):
        try:
            obj = iter(obj)
        except TypeError:
            return False
        else:
            return True

    if isinstance(values1, (int, float)) and isinstance(values2, (int, float)):
        return values1 * values2

    if isiterator(values1) and not isiterator(values2):
        return Dot(values1, [values2]*len(values1))

    if not isiterator(values1) and isiterator(values2):
        return Dot([values1]*len(values2), values2)

    return Sum(va1 * va2 for va1, va2 in zip(values1, values2))
