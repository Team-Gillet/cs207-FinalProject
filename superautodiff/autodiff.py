from collections import Counter


class AutoDiff():
    """Creates an object for autodifferentiation

    ATTRIBUTES
    ==========
    var : name of variables
    val : the value of the object
    der : the derivative of the object

    EXAMPLES
    ========
    >>> x = AutoDiff("x",4)
    >>> x.var
    {"x"}
    >>> x.val
    4
    >>> x.der
    Counter({'x': 1.0})
    """

    def __init__(self, var, val, der=1.0):
        if type(var) != set:
            self.var = set(var)
        else:
            self.var = var
        self.val = val
        if type(der) != float:
            self.der = der
        else:
            self.der = Counter({var: der})

    def __add__(self, other):
        try:  # ask forgiveness
            total = Counter()
            total.update(self.der)
            total.update(other.der)
            var = self.var | other.var
            return AutoDiff(var, self.val + other.val, total)
        except AttributeError:
            return AutoDiff(self.var, self.val + other, self.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            total = Counter()
            total.update(self.der)
            total.update((-other).der)
            var = self.var | other.var
            return AutoDiff(var, self.val - other.val, total)
        except AttributeError:
            return AutoDiff(self.var, self.val - other.val, self.der)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        try:
            total = Counter()
            der1 = {k: other.val * v for k, v in self.der.items()}
            der2 = {k: self.val * v for k, v in other.der.items()}
            total.update(der1)
            total.update(der2)
            var = self.var | other.var
            return AutoDiff(var, self.val * other.val, total)
        except AttributeError:
            der1 = {k: other * v for k, v in self.der.items()}
            return AutoDiff(self.var, self.val * other, der1)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        neg = {k: -1 * v for k, v in self.der.items()}
        return AutoDiff(self.var, -self.val, neg)

    def reciprocal(self):
        value = -1 / (self.val * self.val)
        der = {k: value * v for k, v in self.der.items()}
        return AutoDiff(self.var, 1 / self.val, der)

    def __truediv__(self, other):
        try:
            return self * other.reciprocal()
        except AttributeError:
            return AutoDiff(self.var, self.val / other, self.der)

    def __rtruediv__(self, other):
        return self.__rtruediv__(other)

    def __pow__(self, power):
        try:
            value = power * (self.val) ** (power - 1)
            der = {k: value * v for k, v in self.der.items()}
            return AutoDiff(self.var, self.val ** power, der)
        except AttributeError:
            return AutoDiff(self.var, self.val ** power, self.der)
