import numpy as np
import math
from superautodiff.autodiff import AutoDiff

def sin(x):
    try:
        val = np.sin(x.val)
        der = {k: np.cos(x.val) * v for k, v in x.der.items()}
        return AutoDiff(x.var, val, der)
    except AttributeError:
        return np.sin(x)


def cos(x):
    try:
        val = np.cos(x.val)
        der = {k: -np.sin(x.val) * v for k, v in x.der.items()}
        return AutoDiff(x.var, val, der)
    except AttributeError:
        return np.cos(x)


def tan(x):
    try:
        val = np.tan(x.val)
        der = {k: (1 / (np.cos(x.val) ** 2)) * v for k, v in x.der.items()}
        return AutoDiff(x.var, val, der)
    except AttributeError:
        return np.tan(x)


def arcsin(x):
    try:
        val = np.arcsin(x.val)
        der = {k: (1 / np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        return AutoDiff(x.var, val, der)
    except AttributeError:
        return np.arcsin(x)


def arccos(x):
    try:
        val = np.arccos(x.val)
        der = {k: (1 / -np.sqrt(1 - x.val ** 2)) * v for k, v in x.der.items()}
        return AutoDiff(x.var, val, der)
    except AttributeError:
        return np.arccos(x)


def arctan(x):
    try:
        val = np.arctan(x.val)
        der = {k: (1 / (1 + x.val * x.val)) * v for k, v in x.der.items()}
        return AutoDiff(x.var, val, der)
    except AttributeError:
        return np.arctan(x)


def exp(x):
    try:
        val = np.exp(x.val)
        der = {k: val * v for k, v in x.der.items()}
        return AutoDiff(x.var, val, der)
    except AttributeError:
        return np.exp(x)


def log(x, base=math.e):
    try:
        val = math.log(x.val, base)
        der = {k: (1 / (x.val * math.log(base))) * v for k, v in x.der.items()}
        return AutoDiff(x.var, val, der)
    except ValueError:
        print("Unvalid Value")
    except AttributeError:
        return math.log(x, base)

## sinh(x),cosh(x)...
