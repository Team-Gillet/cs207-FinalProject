import numpy as np
import math
import superautodiff as sad

def sin(x):

    if (type(x).__name__) is 'AutoDiffVector':
        return _sinV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _sinR(x)

    try:
        var = x.var
        val = np.sin(x.val)
        der = {k: np.cos(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: sinV(ADV) instead of sin(ADV)")
        return np.sin(x)

def _sinV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = sin(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: sin(AD) instead of sinV(AD)")
        return np.sin(x)

def _sinR(x):
    der = {x.var : np.cos(x.val)}
    return sad.AutoDiffReverse(np.sin(x.val), None, der)

def cos(x):

    if (type(x).__name__) is 'AutoDiffVector':
        return _cosV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _cosR(x)

    try:
        var = x.var
        val = np.cos(x.val)
        der = {k: -np.sin(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: cosV(ADV) instead of cos(ADV)")
        return np.cos(x)

def _cosV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = cos(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: cos(AD) instead of cosV(AD)")
        return np.cos(x)

def _cosR(x):
    der = {x.var : -np.sin(x.val)}
    return sad.AutoDiffReverse(np.cos(x.val), None, der)

def tan(x):

    if (type(x).__name__) is 'AutoDiffVector':
        return _tanV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _tanR(x)

    try:
        var = x.var
        val = np.tan(x.val)
        der = {k: (1 / (np.cos(x.val) ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: tanV(ADV) instead of tan(ADV)")
        return np.tan(x)

def _tanV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = tan(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: tan(AD) instead of tanV(AD)")
        return np.tan(x)

def _tanR(x):
    der = {x.var : 1 / (np.cos(x.val) ** 2)}
    return sad.AutoDiffReverse(np.tan(x.val), None, der)

def arcsin(x):

    if (type(x).__name__) is 'AutoDiffVector':
        return _arcsinV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _arcsinR(x)

    try:
        var = x.var
        val = np.arcsin(x.val)
        der = {k: (1 / np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: arcsinV(ADV) instead of arcsin(ADV)")
        return np.arcsin(x)

def _arcsinV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = arcsin(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: arcsin(AD) instead of arcsinV(AD)")
        return np.arcsin(x)

def _arcsinR(x):
    der = {x.var : 1 / np.sqrt(1 - x.val ** 2)}
    return sad.AutoDiffReverse(np.arcsin(x.val), None, der)

def arccos(x):

    if (type(x).__name__) is 'AutoDiffVector':
        return _arccosV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _arccosR(x)

    try:
        var = x.var
        val = np.arccos(x.val)
        der = {k: (1 / -np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: arccosV(ADV) instead of arccos(ADV)")
        return np.arccos(x)


def _arccosV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = arccos(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: arccos(AD) instead of arccosV(AD)")
        return np.arccos(x)

def _arccosR(x):
    der = {x.var : 1 / -np.sqrt(1 - x.val ** 2)}
    return sad.AutoDiffReverse(np.arccos(x.val), None, der)

def arctan(x):
    
    if (type(x).__name__) is 'AutoDiffVector':
        return _arctanV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _arctanR(x)

    try:
        var = x.var
        val = np.arctan(x.val)
        der = {k: (1 / (1 + x.val * x.val)) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: arctanV(ADV) instead of arctan(ADV)")
        return np.arctan(x)


def _arctanV(x):

    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = arctan(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: arctan(AD) instead of arctanV(AD)")
        return np.arctan(x)

def _arctanR(x):
    der = {x.var : 1 / (1 + x.val * x.val)}
    return sad.AutoDiffReverse(np.arctan(x.val), None, der)

def exp(x):
    
    if (type(x).__name__) is 'AutoDiffVector':
        return _expV(x)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _expR(x)

    try:
        var = x.var
        val = np.exp(x.val)
        der = {k: val * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: expV(ADV) instead of exp(ADV)")
        return np.exp(x)

def _expV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = exp(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: exp(AD) instead of expV(AD)")
        return np.exp(x)

def _expR(x):
    der = {x.var : np.exp(x.val)}
    return sad.AutoDiffReverse(np.exp(x.val), None, der)

def log(x, base=math.e):

    if (type(x).__name__) is 'AutoDiffVector':
        return _logV(x, base=base)
    elif (type(x).__name__) is 'AutoDiffReverse':
        return _logR(x)

    try:
        var = x.var
        val = math.log(x.val, base)
        der = {k: (1 / (x.val * math.log(base))) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: logV(ADV) instead of log(ADV)")
        return math.log(x, base)

def _logV(x, base=math.e):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = log(x.objects[list(x.objects.keys())[i]], base=base)
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: log(AD) instead of logV(AD)")
        return np.log(x)

def _logR(x, base=math.e):
    der = {x.var : 1 / (x.val * math.log(base))}
    return sad.AutoDiffReverse(math.log(x.val, base), None, der)

def sinh(x):

    if (type(x).__name__) is 'AutoDiffVector':
        return _sinhV(x)


    try:
        var = x.var
        val = math.sinh(x.val)
        der = {k: math.cosh(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: sinhV(ADV) instead of sinh(ADV)")
        return math.sinh(x)


def _sinhV(x):
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
    
    if (type(x).__name__) is 'AutoDiffVector':
        return _coshV(x)

    try:
        var = x.var
        val = math.cosh(x.val)
        der = {k: math.sinh(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: coshV(ADV) instead of cosh(ADV)")
        return math.cosh(x)

def _coshV(x):
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

    if (type(x).__name__) is 'AutoDiffVector':
        return _tanhV(x)
    
    try:
        var = x.var
        val = math.tanh(x.val)
        der = {k: (1 / (cosh(x.val) ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        # print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: tanhV(ADV) instead of tanh(ADV)")
        return np.tanh(x)

def _tanhV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = tanh(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: tanh(AD) instead of tanhV(AD)")
        return np.tanh(x)

    
def sqrt(x):
  try:
    return x**0.5
  except:
    return np.sqrt(x)

def logistic(x):
  return 1/(1+exp(-x))
