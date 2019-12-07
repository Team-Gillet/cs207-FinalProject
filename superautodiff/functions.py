import numpy as np
import math
import superautodiff as sad

def sin(x):
    try:
        val = np.sin(x.val)
        der = {k: np.cos(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: sinV(ADV) instead of sin(ADV)")
        return np.sin(x)

def sinV(x):
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
    try:
        val = np.cos(x.val)
        der = {k: -np.sin(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: cosV(ADV) instead of cos(ADV)")
        return np.cos(x)

def cosV(x):
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
    try:
        val = np.tan(x.val)
        der = {k: (1 / (np.cos(x.val) ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: tanV(ADV) instead of tan(ADV)")
        return np.tan(x)

def tanV(x):
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
    try:
        val = np.arcsin(x.val)
        der = {k: (1 / np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: arcsinV(ADV) instead of arcsin(ADV)")
        return np.arcsin(x)

def arcsinV(x):
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
    try:
        val = np.arccos(x.val)
        der = {k: (1 / -np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: arccosV(ADV) instead of arccos(ADV)")
        return np.arccos(x)


def arccosV(x):
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
    try:
        val = np.arctan(x.val)
        der = {k: (1 / (1 + x.val * x.val)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: arctanV(ADV) instead of arctan(ADV)")
        return np.arctan(x)

def arctanV(x):
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
    try:
        val = np.exp(x.val)
        der = {k: val * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: expV(ADV) instead of exp(ADV)")
        return np.exp(x)

def expV(x):
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
    try:
        val = math.log(x.val, base)
        der = {k: (1 / (x.val * math.log(base))) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: logV(ADV) instead of log(ADV)")
        return math.log(x, base)

def logV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = log(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: log(AD) instead of logV(AD)")
        return np.log(x)

def sinh(x):
    try:
        val = math.sinh(x.val)
        der = {k: math.cosh(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: sinhV(ADV) instead of sinh(ADV)")
        return math.sinh(x)


def sinhV(x):
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
    try:
        val = math.cosh(x.val)
        der = {k: math.sinh(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: coshV(ADV) instead of cosh(ADV)")
        return math.cosh(x)

def coshV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = cosh(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: cosh(AD) instead of coshV(AD)")
        return np.cosh(x)


def tanh(x):
    try:
        val = math.tanh(x.val)
        der = {k: (1 / (cosh(x.val) ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except ValueError:
        print("Invalid value for mathematical function")
    except AttributeError:
        print("Warning: For AutoDiffVector objects, please use the corresponding mathematical function: tanhV(ADV) instead of tanh(ADV)")
        return 1 / (cosh(x) ** 2)

def tanhV(x):
    try:
        objects = []
        for i in range(len(x.objects)):
            new_object = tanh(x.objects[list(x.objects.keys())[i]])
            objects.append(new_object)
            
        return sad.autodiff.AutoDiffVector(objects)
    except AttributeError:
        print("Warning: For AutoDiff objects, please use the corresponding mathematical function: tanh(AD) instead of tanhV(AD)")
        return np.tanh(x)


if __name__ == '__main__':
    x1 = sad.AutoDiff('x',1)