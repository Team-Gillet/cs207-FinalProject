import numpy as np
import superautodiff as sad

def cos(x):
    der = {x.var : -np.sin(x.val)}
    return sad.AutoDiffReverse(np.cos(x.val), None, der)


def sin(x):
    der = {x.var : np.cos(x.val)}
    return sad.AutoDiffReverse(np.sin(x.val), None, der)
