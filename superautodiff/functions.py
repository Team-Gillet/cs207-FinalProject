import numpy as np
import math
import superautodiff as sad
from collections import Counter

def sin(x):
    """Returns the sine of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _sinV(x)

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
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = sin(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: sin(AD) instead of sinV(AD)")
        return np.sin(x)

def cos(x):
    """Returns the cosine of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _cosV(x)


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
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = cos(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: cos(AD) instead of cosV(AD)")
        return np.cos(x)

def tan(x):
    """Returns the tangent of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _tanV(x)

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
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = tan(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: tan(AD) instead of tanV(AD)")
        return np.tan(x)


def arcsin(x):
    """Returns the arcsine of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _arcsinV(x)

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
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = arcsin(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: arcsin(AD) instead of arcsinV(AD)")
        return np.arcsin(x)

def arccos(x):
    """Returns the arccos of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _arccosV(x)

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
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = arccos(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: arccos(AD) instead of arccosV(AD)")
        return np.arccos(x)


def arctan(x):
    """Returns the arctan of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""
    
    if (type(x).__name__) is 'AutoDiffVector':
        return _arctanV(x)

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
    """Returns the arctan of the AutoDiffVector object"""

    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = arctan(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: arctan(AD) instead of arctanV(AD)")
        return np.arctan(x)


def exp(x):
    """Returns the exp of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""
    
    if (type(x).__name__) is 'AutoDiffVector':
        return _expV(x)

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

    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = exp(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: exp(AD) instead of expV(AD)")
        return np.exp(x)

def log(x, base=math.e):
    """Returns the log of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _logV(x, base=base)

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

    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = log(x.objects[list(x.objects.keys())[i]], base=base)
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: log(AD) instead of logV(AD)")
        return np.log(x)

def sinh(x):
    """Returns the sine_h of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _sinhV(x)


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

    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = sinh(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: sinh(AD) instead of sinhV(AD)")
        return np.sinh(x)


def cosh(x):
    """Returns the cosine_h of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""
    
    if (type(x).__name__) is 'AutoDiffVector':
        return _coshV(x)

    try:
        var = x.var
        val = math.cosh(x.val)
        der = {k: math.sinh(x.val) * v for k, v in x.der.items()}
        der = Counter(der)
        return sad.AutoDiff(var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: coshV(ADV) instead of cosh(ADV)")
        return math.cosh(x)

def _coshV(x):
    """Returns the cosine_h of the AutoDiffVector object"""
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = cosh(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        # print("Warning: For AutoDiff objects, please use the corresponding mathematical function: cosh(AD) instead of coshV(AD)")
        return np.cosh(x)


def tanh(x):
    """Returns the tan_h of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""

    if (type(x).__name__) is 'AutoDiffVector':
        return _tanhV(x)
    
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
    
def sqrt(x):
    """Returns the square root of the AutoDiff or AutoDiffVector or AutoDiffReverse object"""
  try:
    return x**0.5
  except:
    return np.sqrt(x)

def logistic(x):
    """Returns the AutoDiff or AutoDiffVector or AutoDiffReverse object passed through a sigmoid transformation"""
  return 1/(1+exp(-x))

def _tanhV(x):
    """Returns the tan_h of the AutoDiffVector object"""
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = tanh(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: tanh(AD) instead of tanhV(AD)")
        return np.tanh(x)
