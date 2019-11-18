import numpy as np
import math
import superautodiff as sad

def sin(x):
    try:
        val = np.sin(x.val)
        der = {k: np.cos(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        return np.sin(x)


def cos(x):
    try:
        val = np.cos(x.val)
        der = {k: -np.sin(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        return np.cos(x)


def tan(x):
    try:
        val = np.tan(x.val)
        der = {k: (1 / (np.cos(x.val) ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        return np.tan(x)


def arcsin(x):
    try:
        val = np.arcsin(x.val)
        der = {k: (1 / np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        return np.arcsin(x)


def arccos(x):
    try:
        val = np.arccos(x.val)
        der = {k: (1 / -np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        return np.arccos(x)


def arctan(x):
    try:
        val = np.arctan(x.val)
        der = {k: (1 / (1 + x.val * x.val)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        return np.arctan(x)


def exp(x):
    try:
        val = np.exp(x.val)
        der = {k: val * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except AttributeError:
        return np.exp(x)


def log(x, base=math.e):
    try:
        val = math.log(x.val, base)
        der = {k: (1 / (x.val * math.log(base))) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except ValueError:
        print("Unvalid Value")
    except AttributeError:
        return math.log(x, base)


def sinh(x):
    try:
        val = math.sinh(x.val)
        der = {k: math.cosh(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except ValueError:
        print("Unvalid Value")
    except AttributeError:
        return math.sinh(x)


def cosh(x):
    try:
        val = math.cosh(x.val)
        der = {k: math.sinh(x.val) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except ValueError:
        print("Unvalid Value")
    except AttributeError:
        return math.cosh(x)


def tanh(x):
    try:
        val = math.tanh(x.val)
        der = {k: (1 / (cosh(x.val) ** 2)) * v for k, v in x.der.items()}
        return sad.AutoDiff(x.var, val, der)
    except ValueError:
        print("Unvalid Value")
    except AttributeError:
        return 1 / (cosh(x) ** 2)

if __name__ == '__main__':
    x1 = sad.AutoDiff('x',1)