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
        #add exception for using same variable name?
        if type(var) == int or type(var) == float:
            raise ValueError("Input variable should be a string or a set or string")
        elif type(var) == str:
            self.var = set(var)
        else:
            self.var = var
        if type(val)==list or type(val)==str:
            raise ValueError("Input value should be integer or float")
        else:
            self.val = float(val)
        #if type(val)==float or type(val)==int:
        #    self.val = float(val)
        #else:
        #    raise ValueError("Input value should be integer or float")
        if type(der) != float:
            self.der = der
        else:
            self.der = Counter({var: der})

    def __add__(self, other):
    	"""Performs addition on two AutoDiff objects"""

        try:  # ask forgiveness
            total = Counter()
            total.update(self.der)
            total.update(other.der)
            var = self.var | other.var
            return AutoDiff(var, self.val + other.val, total)
        except AttributeError:
            return AutoDiff(self.var, self.val + other, self.der)

    def __radd__(self, other):
    	"""Performs addition on two AutoDiff objects"""

        return self.__add__(other)

    def __sub__(self, other):
        """Performs subtraction on two AutoDiff objects"""

        try:
            total = Counter()
            total.update(self.der)
            total.update((-other).der)
            var = self.var | other.var
            return AutoDiff(var, self.val - other.val, total)
        except AttributeError:
            return AutoDiff(self.var, self.val - other, self.der)

    def __rsub__(self, other):
    	"""Performs subtraction on two AutoDiff objects"""

        return self.__sub__(other)

    def __mul__(self, other):
    	"""Performs multiplication of an AutoDiff object with scalars and other AutoDiff objects"""

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
    	"""Performs multiplication of an AutoDiff object with scalars and other AutoDiff objects"""

        return self.__mul__(other)

    def __neg__(self):
    	"""Returns the negation of an AutoDiff object"""

        neg = {k: -1 * v for k, v in self.der.items()}
        return AutoDiff(self.var, -self.val, neg)

    def reciprocal(self):
    	"""Returns the reciprocal of an AutoDiff object"""

        value = -1 / (self.val * self.val)
        der = {k: value * v for k, v in self.der.items()}
        return AutoDiff(self.var, 1 / self.val, der)

    def __truediv__(self,other):
    	"""Performs division of an AutoDiff object with scalars and other AutoDiff objects"""
        try:
            return self*other.reciprocal()
        except AttributeError:
            der = {k:v/other for k, v in self.der.items()}
            return AutoDiff(self.var,self.val/other,der)

    def __rtruediv__(self, other): 
    	"""Performs division of an AutoDiff object with scalars and other AutoDiff objects"""
      	#x._rtruediv_(other) <==> other / x
      	return other*self.reciprocal()

    def __pow__(self, power):
    	"""Performs exponentiation of an AutoDiff object with scalars values"""
        try:
            value = power * (self.val) ** (power - 1)
            der = {k: value * v for k, v in self.der.items()}
            return AutoDiff(self.var, self.val ** power, der)
        except AttributeError:
            return AutoDiff(self.var, self.val ** power, self.der)

    def __eq__(self, other):
    	"""Assesses the equality of two AutoDiff objects"""
    	return self.val == other.val and self.der == other.der

    def __ne__(self, other):
    	"""Assesses the equality of two AutoDiff objects"""
    	return not self.val == other.val and self.der == other.der

    def  __lt__(self, other):
    	"""Assesses whether an AutoDiff object value is less than that of another AutoDiff object"""
    	return self.val < other.val

    def  __le__():
    	"""Assesses whether an AutoDiff object value is less than or equal to that of another AutoDiff object"""
    	return self.val <= other.val

    def __ge__():
    	"""Assesses whether an AutoDiff object value is greater than or equal to that of another AutoDiff object"""
    	return self.val >= other.val

    def __gt__(self, other):
    	"""Assesses whether an AutoDiff object value is greater than that of another AutoDiff object"""
    	return self.val > other.val

