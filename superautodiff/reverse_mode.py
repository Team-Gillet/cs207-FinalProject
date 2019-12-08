import pandas as pd
import numpy as np

index = 0
forward_pass = pd.DataFrame(columns=['Node', 'd1', 'd1value', 'd2', 'd2value'])


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

    def __init__(self, val, var=None, der=1.0):
        global forward_pass
        if not var:
            global index
            var = 'y' + str(index + 1)
            index += 1
            self.var = var
        else:
            self.var = var
        if type(val) == list or type(val) == str:
            raise ValueError("Input value should be integer or float")
        else:
            self.val = float(val)
        if type(der) != float:
            self.der = der
            if len(list(der.keys())) > 1:
                row = pd.DataFrame(
                    [[var, list(der.keys())[0], list(der.values())[0], list(der.keys())[1], list(der.values())[1]]],
                    columns=['Node', 'd1', 'd1value', 'd2', 'd2value'])
                forward_pass = forward_pass.append(row)
            else:
                row = pd.DataFrame([[var, list(der.keys())[0], list(der.values())[0], '-', '-']],
                                   columns=['Node', 'd1', 'd1value', 'd2', 'd2value'])
                forward_pass = forward_pass.append(row)
        else:
            print(der)
            print(var)
            self.der = {var: 1}
            row = pd.DataFrame([[var, var, 1, '-', '-']], columns=['Node', 'd1', 'd1value', 'd2', 'd2value'])
            forward_pass = forward_pass.append(row)

    def __add__(self, other):
        """Performs addition on two AutoDiff objects"""
        try:
            total = {self.var: 1, other.var: 1}
            return AutoDiff(self.val + other.val, None, der=total)
        except AttributeError:
            return AutoDiff(self.val + other, None, {self.var: 1})

    def __radd__(self, other):
        """Performs addition on two AutoDiff objects"""
        return self.__add__(other)

    def __sub__(self, other):
        """Performs subtraction on two AutoDiff objects"""
        try:
            total = {self.var: 1, other.var: -1}
            return AutoDiff(self.val - other.val, None, der=total)
        except AttributeError:
            return AutoDiff(self.val - other, None, {self.var: 1})

    def __rsub__(self, other):
        """Performs subtraction on two AutoDiff objects"""
        try:
            total = {self.var: 1, other.var: -1}
            return AutoDiff(self.val - other.val, None, der=total)
        except AttributeError:
            return AutoDiff(self.val - other, None, {self.var: -1})

    def __mul__(self, other):
        """Performs subtraction on two AutoDiff objects"""
        try:
            total = {self.var: other.val, other.var: self.val}
            return AutoDiff(self.val * other.val, None, total)
        except AttributeError:
            return AutoDiff(self.val * other, None, {self.var: other})

    def __rmul__(self, other):
        """Performs subtraction on two AutoDiff objects"""
        return self.__mul__(other)

    def __neg__(self):
        """Returns the negation of an AutoDiff object"""
        return 0 - self

    def __pow__(self, power):
        """Performs exponentiation of an AutoDiff object with scalars values e.g x**3 """
        value = power * (self.val) ** (power - 1)
        der = {k: value * v for k, v in self.der.items()}
        return AutoDiff(self.val ** power, None, der)

    def __rpow__(self, power):
        """Performs exponentiation of an AutoDiff object with scalars values e.g. 3**x"""
        value = power ** self.val
        der = {k: value * v * np.log(power) for k, v in self.der.items()}
        return AutoDiff(value, None, der)

    def reciprocal(self):
        """Returns the reciprocal of an AutoDiff object"""
        value = -1 / (self.val * self.val)
        der = {k: value * v for k, v in self.der.items()}
        return AutoDiff(1 / self.val, None, der)

    def __truediv__(self, other):
        """Performs division of an AutoDiff object with scalars and other AutoDiff objects"""
        try:
            value = -1 / (self.val * self.val)
            der = {k: value * v for k, v in self.der.items()}
            total = {self.var: other.val, other.var: self.val}
            return AutoDiff(self.val * other.val, None, total)
            return AutoDiff(1 / self.val, None, der)
            return self * other.reciprocal()
        except AttributeError:
            der = {k: v / other for k, v in self.der.items()}
            print('truediv')
            return AutoDiff(self.val / other, None, der)

    def __rtruediv__(self, other):
        """Performs division of an AutoDiff object with scalars and other AutoDiff objects"""
        # x._rtruediv_(other) <==> other / x
        print('rtruediv')
        return other * self.reciprocal()