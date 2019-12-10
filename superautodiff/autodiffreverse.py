import pandas as pd
import numpy as np

forward_pass_index = 0
forward_pass = pd.DataFrame(columns=['Node', 'd1', 'd1value', 'd2', 'd2value'])


class AutoDiffReverse():
    """Creates an object for AutoDiffReverseerentiation

    ATTRIBUTES
    ==========
    var : name of variables
    val : the value of the object
    der : the derivative of the object

    EXAMPLES
    ========
    >>> x = AutoDiffReverse(4, 'x')
    >>> x.var
    'x'
    >>> x.val
    4.0
    >>> x.der
    {'x': 1}
    """

    def __init__(self, val, var=None, der=1.0):
        global forward_pass
        if not var:
            global forward_pass_index
            var = 'y' + str(forward_pass_index + 1)
            forward_pass_index += 1
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
            self.der = {var: 1}
            row = pd.DataFrame([[var, var, 1, '-', '-']], columns=['Node', 'd1', 'd1value', 'd2', 'd2value'])
            forward_pass = forward_pass.append(row)

    def __add__(self, other):
        """Performs addition on two AutoDiffReverse objects"""
        try:
            total = {self.var: 1, other.var: 1}
            return AutoDiffReverse(self.val + other.val, None, der=total)
        except AttributeError:
            return AutoDiffReverse(self.val + other, None, {self.var: 1})

    def __radd__(self, other):
        """Performs addition on two AutoDiffReverse objects"""
        return self.__add__(other)

    def __sub__(self, other):
        """Performs subtraction on two AutoDiffReverse objects"""
        try:
            total = {self.var: 1, other.var: -1}
            return AutoDiffReverse(self.val - other.val, None, der=total)
        except AttributeError:
            return AutoDiffReverse(self.val - other, None, {self.var: 1})

    def __rsub__(self, other):
        """Performs subtraction on two AutoDiffReverse objects"""
        try:
            total = {self.var: 1, other.var: -1}
            return AutoDiffReverse(self.val - other.val, None, der=total)
        except AttributeError:
            return AutoDiffReverse(self.val - other, None, {self.var: -1})

    def __mul__(self, other):
        """Performs subtraction on two AutoDiffReverse objects"""
        try:
            total = {self.var: other.val, other.var: self.val}
            return AutoDiffReverse(self.val * other.val, None, total)
        except AttributeError:
            return AutoDiffReverse(self.val * other, None, {self.var: other})

    def __rmul__(self, other):
        """Performs subtraction on two AutoDiffReverse objects"""
        return self.__mul__(other)

    def __neg__(self):
        """Returns the negation of an AutoDiffReverse object"""
        return 0 - self

    def __pow__(self, power):
        """Performs exponentiation of an AutoDiffReverse object with scalars values e.g x**3 """
        value = power * (self.val) ** (power - 1)
        der = {k: value * v for k, v in self.der.items()}
        return AutoDiffReverse(self.val ** power, None, der)

    def __rpow__(self, power):
        """Performs exponentiation of an AutoDiffReverse object with scalars values e.g. 3**x"""
        value = power ** self.val
        der = {k: value * v * np.log(power) for k, v in self.der.items()}
        return AutoDiffReverse(value, None, der)

    def __truediv__(self, other):
        """Performs division of an AutoDiffReverse object with scalars and other AutoDiffReverse objects"""
        try:
            value = -1 / (other.val * other.val)
            total = {self.var: 1 / other.val, other.var: value * self.val}
            return AutoDiffReverse(self.val / other.val, None, total)
        except AttributeError:
            total = {self.var: 1 / other}
            return AutoDiffReverse(self.val / other, None, total)

    def __rtruediv__(self, other):
        """Performs division of an AutoDiffReverse object with scalars and other AutoDiffReverse objects"""
        value = -1 / (self.val * self.val)
        total = {self.var: other * value}
        return AutoDiffReverse(other / self.val, None, total)
    
    def pass_table(self):
        return forward_pass
    
    def clear_table(self):
        forward_pass = pd.DataFrame(columns=['Node', 'd1', 'd1value', 'd2', 'd2value'])
    
# Reverse Pass  
def reversepass(df,vars):
  le = len(vars)
  df["currentvalue"] = df[['d1', 'd2']].apply(lambda x: ''.join(x), axis=1)
  dic = {}
  dic[df.iloc[-1]["Node"]] =1
  ### start from the second last node to first
  for l in range(-2,-(df.shape[0]+1),-1):
    n = df.iloc[l]["Node"]
    der = 0
  ###  scan all the rows  
    for row in range(le,df.shape[0]):
      if n in df.iloc[row]["currentvalue"]:
        parent = df.iloc[row]["Node"]
        if n == df.iloc[row]["d1"]:
          der_n = df.iloc[row]["d1value"]
        elif n == df.iloc[row]["d2"]:
          der_n = df.iloc[row]["d2value"]
        der = der + dic[parent]*der_n
    dic[n] = der
  dic_out = dict((k, dic[k]) for k in vars) 
  return dic_out
