import numpy as np
import math
import superautodiff as sad

def cos(x):
    der = {x.var : -np.sin(x.val)}
    return AutoDiff(np.cos(x.val), None, der)


def sin(x):
    der = {x.var : np.cos(x.val)}
    return AutoDiff(np.sin(x.val), None, der)


def tan(x):
    der = {x.var : 1 / (np.cos(x.val) ** 2)}
    return AutoDiff(np.tan(x.val), None, der)


def arcsin(x):
    der = {x.var : 1 / np.sqrt(1 - x.val ** 2)}
    return AutoDiff(np.arcsin(x.val), None, der)


def arccos(x):
    der = {x.var : 1 / -np.sqrt(1 - x.val ** 2)}
    return AutoDiff(np.arccos(x.val), None, der)


def arctan(x):
    der = {x.var : 1 / (1 + x.val * x.val)}
    return AutoDiff(np.arctan(x.val), None, der)


def exp(x):
    der = {x.var : np.exp(x.val)}
    return AutoDiff(np.exp(x.val), None, der)


def log(x, base=math.e):
    der = {x.var : 1 / (x.val * math.log(base))}
    return AutoDiff(math.log(x.val, base), None, der)
