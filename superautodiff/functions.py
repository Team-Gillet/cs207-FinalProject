import numpy as np
import math
import superautodiff as sad
from collections import Counter

def jacobian(variables, functions):
    """Returns the Jacobian matrix containing the first derivative of each input function with respect to each input variable"""
    derivatives = []
    
    try:
        # Case where functions is a list of AD objects
        if type(functions) is list:
            
            if len(functions) is 0:
                raise ValueError("Functions cannot be empty; input either an AutoDiffVector object or a list of AutoDiff objects")
            
            for function in functions:
                for variable in variables:
                    derivatives.append(function.der[variable])
            
            return np.array(derivatives).reshape(len(functions), len(variables))
        
        # Case where functions is an ADV object
        else:
            if len(functions.objects) is 0:
                raise ValueError("Functions cannot be empty; input either an AutoDiffVector object or a list of AutoDiff objects")
            
            try:
                for key in list(functions.objects.keys()):
                    for variable in variables:
                        derivatives.append((functions.objects[key].der[variable]))
                return np.array(derivatives).reshape(len(functions.objects), len(variables))
            
            # If neither case fits raise an error
            except:
                raise ValueError("Function inputs need to either be an AutoDiffVector object or an array of AutoDiff objects")
    except:
        raise ValueError("Incorrect input")
                    
                


def sin(x):
    """Returns the sine of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _sinV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _sinR(x)

    try:
        var = x.var
        val = np.sin(x.val)
        der = {k: np.cos(x.val) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: sinV(ADV) instead of sin(ADV)")
        return np.sin(x)

def _sinV(x):
    """Returns the sine of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = sin(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _sinR(x):
    """Returns the sine of the AutoDiffReverse object"""
    der = {x.var : np.cos(x.val)}
    der = Counter(der)
    return sad.AutoDiffReverse(np.sin(x.val), None, der)

def cos(x):
    """Returns the cosine of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _cosV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _cosR(x)

    try:
        var = x.var
        val = np.cos(x.val)
        der = {k: -np.sin(x.val) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: cosV(ADV) instead of cos(ADV)")
        return np.cos(x)

def _cosV(x):
    """Returns the cosine of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = cos(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _cosR(x):
    """Returns the cosine of the AutoDiffReverse object"""

    der = {x.var : -np.sin(x.val)}
    der = Counter(der)
    return sad.AutoDiffReverse(np.cos(x.val), None, der)

def tan(x):
    """Returns the tangent of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _tanV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _tanR(x)

    try:
        var = x.var
        val = np.tan(x.val)
        der = {k: (1 / (np.cos(x.val) ** 2)) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: tanV(ADV) instead of tan(ADV)")
        return np.tan(x)

def _tanV(x):
    """Returns the tangent of the AutoDiffVectorobject"""

    objects = []
    for i in range(len(x.objects)):
        new_object = tan(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _tanR(x):
    """Returns the rangent of the AutoDiffReverse object"""

    der = {x.var : 1 / (np.cos(x.val) ** 2)}
    return sad.AutoDiffReverse(np.tan(x.val), None, der)

def arcsin(x):
    """Returns the arcsine of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _arcsinV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _arcsinR(x)

    try:
        var = x.var
        val = np.arcsin(x.val)
        der = {k: (1 / np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: arcsinV(ADV) instead of arcsin(ADV)")
        return np.arcsin(x)

def _arcsinV(x):
    """Returns the arcsine of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = arcsin(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _arcsinR(x):
    """Returns the arcsine of the AutoDiffReverse object"""

    der = {x.var : 1 / np.sqrt(1 - x.val ** 2)}
    return sad.AutoDiffReverse(np.arcsin(x.val), None, der)

def arccos(x):
    """Returns the arccos of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _arccosV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _arccosR(x)

    try:
        var = x.var
        val = np.arccos(x.val)
        der = {k: (1 / -np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: arccosV(ADV) instead of arccos(ADV)")
        return np.arccos(x)


def _arccosV(x):
    """Returns the arccos of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = arccos(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _arccosR(x):
    """Returns the arccsine of the AutoDiffReverse object"""

    der = {x.var : 1 / -np.sqrt(1 - x.val ** 2)}
    return sad.AutoDiffReverse(np.arccos(x.val), None, der)

def arctan(x):
    """Returns the arctangent of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""
    
    if (type(x).__name__) is 'AutoDiffVector':
        return _arctanV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _arctanR(x)

    try:
        var = x.var
        val = np.arctan(x.val)
        der = {k: (1 / (1 + x.val * x.val)) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: arctanV(ADV) instead of arctan(ADV)")
        return np.arctan(x)


def _arctanV(x):
    """Returns the arctangent of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = arctan(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _arctanR(x):
    """Returns the arctangent of the AutoDiffReverse object"""

    der = {x.var : 1 / (1 + x.val * x.val)}
    return sad.AutoDiffReverse(np.arctan(x.val), None, der)

def exp(x):
    """Returns the exp of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""
    
    if (type(x).__name__) is 'AutoDiffVector':
        return _expV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _expR(x)

    try:
        var = x.var
        val = np.exp(x.val)
        der = {k: val * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: expV(ADV) instead of exp(ADV)")
        return np.exp(x)

def _expV(x):
    """Returns the exp of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = exp(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _expR(x):
    """Returns the exp of the AutoDiffReverse object"""

    der = {x.var : np.exp(x.val)}
    return sad.AutoDiffReverse(np.exp(x.val), None, der)

def log(x, base=math.e):
    """Returns the log of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _logV(x, base=base)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _logR(x, base=base)

    try:
        var = x.var
        val = math.log(x.val, base)
        der = {k: (1 / (x.val * math.log(base))) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: logV(ADV) instead of log(ADV)")
        return math.log(x, base)

def _logV(x, base=math.e):
    """Returns the log of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = log(x.objects[list(x.objects.keys())[i]], base=base)
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _logR(x, base=math.e):
    """Returns the log of the AutoDiffReverse object"""

    der = {x.var : 1 / (x.val * math.log(base))}
    return sad.AutoDiffReverse(math.log(x.val, base), None, der)

def sinh(x):
    """Returns the sine_h of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _sinhV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _sinhR(x)


    try:
        var = x.var
        val = math.sinh(x.val)
        der = {k: math.cosh(x.val) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: sinhV(ADV) instead of sinh(ADV)")
        return math.sinh(x)


def _sinhV(x):
    """Returns the sine_h of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = sinh(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _sinhR(x):
    """Returns the sin_h of the AutoDiffReverse object"""
    der = {x.var : np.cosh(x.val)}
    der = Counter(der)
    return sad.AutoDiffReverse(np.sinh(x.val), None, der)

def cosh(x):
    """Returns the cosine_h of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""
    
    if (type(x).__name__) is 'AutoDiffVector':
        return _coshV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _coshR(x)

    try:
        var = x.var
        val = math.cosh(x.val)
        der = {k: -math.sinh(x.val) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: coshV(ADV) instead of cosh(ADV)")
        return math.cosh(x)

def _coshV(x):
    """Returns the cosine_h of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = cosh(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _coshR(x):
    """Returns the cos_h of the AutoDiffReverse object"""
    der = {x.var : -np.sinh(x.val)}
    der = Counter(der)
    return sad.AutoDiffReverse(np.cosh(x.val), None, der)

def tanh(x):
    """Returns the tan_h of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _tanhV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _tanhR(x)
    
    try:
        var = x.var
        val = math.tanh(x.val)
        der = {k: (1 / (cosh(x.val) ** 2)) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: tanhV(ADV) instead of tanh(ADV)")
        return np.tanh(x)

def _tanhV(x):
    """Returns the tan_h of the AutoDiffVector object"""

    objects = []
    for i in range(len(x.objects)):
        new_object = tanh(x.objects[list(x.objects.keys())[i]])
        objects.append(new_object)
        
    return sad.autodiff.AutoDiffVector(objects)

def _tanhR(x):
    """Returns the sin_h of the AutoDiffReverse object"""
    der = {x.var : 1/(cosh(x.val) ** 2)}
    der = Counter(der)
    return sad.AutoDiffReverse(np.tanh(x.val), None, der)
      
def sqrt(x):
  """Returns the square root of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""
  return x**0.5

def logistic(x):
  """Returns the AutoDiff or AutoDiffVector or AutoDiffReverse object passed through a sigmoid transformation"""
  return 1/(1+exp(-x))
