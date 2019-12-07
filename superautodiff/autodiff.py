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
            self.var = var
        # else:
        #     self.var = var
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
            return AutoDiff(self.var, self.val - other, self.der)

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

    def __truediv__(self,other):
        try:
            return self*other.reciprocal()
        except AttributeError:
            der = {k:v/other for k, v in self.der.items()}
            return AutoDiff(self.var,self.val/other,der)

    def __rtruediv__(self, other): 
      #x._rtruediv_(other) <==> other / x
      return other*self.reciprocal()

    def __pow__(self, power):
        try:
            value = power * (self.val) ** (power - 1)
            der = {k: value * v for k, v in self.der.items()}
            return AutoDiff(self.var, self.val ** power, der)
        except AttributeError:
            return AutoDiff(self.var, self.val ** power, self.der)



class AutoDiffVector():
    def __init__(self, objects):
        
        # Ensure we have at least one AutoDiff object
        if len(objects) == 0:
            raise ValueError("AutoDiffVector requires at least one AutoDiff object as input")
        
        self.objects = {}
        self.variables = []
        
        # Create dictionary of variables and the AutoDiff objects
        for i in range(len(objects)):
            self.objects[objects[i].var] = objects[i]
            self.variables.append(objects[i].var)
            
        if len(self.variables) != len(set(self.variables)):
            raise ValueError("Variable names cannot be the same")
        
        
    def __add__(self, other):
        
        objects = []
        for i in range(len(self.objects)):
            new_object = self.objects[list(self.objects.keys())[i]] + other
            objects.append(new_object)
            
        return AutoDiffVector(objects)
    
    def __radd__(self, other):
        
        return self.__add__(other)

    def __sub__(self, other):

        objects = []
        for i in range(len(self.objects)):
            new_object = self.objects[list(self.objects.keys())[i]] - other
            objects.append(new_object)
            
        return AutoDiffVector(objects)

    def __rsub__(self,other):

        return self.__sub__(other)

    def __mul__(self, other):

        objects = []
        for i in range(len(self.objects)):
            new_object = self.objects[list(self.objects.keys())[i]] * other
            objects.append(new_object)
            
        return AutoDiffVector(objects)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):

        objects = []
        for i in range(len(self.objects)):
            new_object = -self.objects[list(self.objects.keys())[i]]
            objects.append(new_object)
            
        return AutoDiffVector(objects)

    def __truediv__(self, other):

        objects = []
        for i in range(len(self.objects)):
            new_object = (self.objects[list(self.objects.keys())[i]])/other
            objects.append(new_object)
            
        return AutoDiffVector(objects)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other):

        objects = []
        for i in range(len(self.objects)):
            new_object = (self.objects[list(self.objects.keys())[i]]) ** other
            objects.append(new_object)
            
        return AutoDiffVector(objects)    


def vectorize(var, val, der=1.0):
    """Function takes in an array of variable names, values, and derivatives and creates an AutoDiffVector object"""

    # Ensure that input vectors are list types
    if type(var) != list or type(val) != list:
        raise TypeError("Input array of variables and values need to be of list type")

    # If der is specified, check that it is a list type
    if der != 1.0 and type(der) != list:
        raise TypeError("If derivatives are specified, derivative array needs to be a list")

    # Ensure that var and val are of the same length
    if len(var) != len(val):
        raise KeyError("Input array of variables and values need to be of the same length")

    # If der is specified, check that it is also of the same length
    if der != 1.0 and len(der) != len(val):
        raise KeyError("If derivatives are specified, derivative array needs to be of the same length as the array of variables")

    # If der is unspecified, create a vector of 1.0 values with length equal to input array
    if der == 1.0:
        der = [1.0] * len(var)
        
    # Ensure that the variable names ar enot repeated
    if len(var) != len(set(var)):
        raise ValueError("Variable names cannot be the same")

    # If everything checks out, create AutoDiff objects and to convert into an AutoDiffVector object
    objects = []

    for i in range(len(var)):
        objects.append(AutoDiff(var[i], val[i], der[i]))
    
    return AutoDiffVector(objects)
