import sys
from collections import Counter
import numpy as np
import pytest
import math

sys.path.append('..')
import superautodiff as sad


## Helper function for running the tests
def do_vector_tests(vec1, vec1_other, vec2, vec2_other, der, der_other, val, val_other):
    # Test AutoDiffVector with constant
    for key1 in vec1.objects.keys():
        #get the key of the .der in the current ad object
        key2 = next(iter(vec1.objects[key1].der))
        assert vec1.objects[key1].der[key2] == pytest.approx(der)
        assert vec1.objects[key1].val == pytest.approx(val)
    # Test AutoDiffVector with other AutoDiff object
    for key1 in vec1_other.objects.keys():
        #get the key of the .der in the current ad object
        key2 = next(iter(vec1_other.objects[key1].der))
        assert vec1_other.objects[key1].der[key2] == pytest.approx(der_other)
        assert vec1_other.objects[key1].val == pytest.approx(val_other)
    # Test vectorize with constant
    for key1 in vec2.objects.keys():
        #get the key of the .der in the current ad object
        key2 = next(iter(vec2.objects[key1].der))
        assert vec2.objects[key1].der[key2] == pytest.approx(der)
        assert vec2.objects[key1].val == pytest.approx(val)
    # Test vectorize with other AutoDiff object
    for key1 in vec2_other.objects.keys():
        #get the key of the .der in the current ad object
        key2 = next(iter(vec2_other.objects[key1].der))
        assert vec2_other.objects[key1].der[key2] == pytest.approx(der_other)
        assert vec2_other.objects[key1].val == pytest.approx(val_other)

# Same helper function but without other autodiff (e.g. for testing additive inverse)
def do_vector_tests_no_other(vec1, vec2, der, val):
    # Test AutoDiffVector
    for key1 in vec1.objects.keys():
        #get the key of the .der in the current ad object
        key2 = next(iter(vec1.objects[key1].der))
        assert vec1.objects[key1].der[key2] == pytest.approx(der)
        assert vec1.objects[key1].val == pytest.approx(val)
    # Test vectorize
    for key1 in vec2.objects.keys():
        #get the key of the .der in the current ad object
        key2 = next(iter(vec2.objects[key1].der))
        assert vec2.objects[key1].der[key2] == pytest.approx(der)
        assert vec2.objects[key1].val == pytest.approx(val)


### Test inputs

def test_vector_input_no_args():
    with pytest.raises(TypeError):
        x1 = sad.AutoDiffVector()

def test_vector_input_two_args():
    with pytest.raises(TypeError):
        x1 = sad.AutoDiffVector('x',1)

def test_vector_inputlist_not_autodiff():
    with pytest.raises(ValueError):
        x = sad.AutoDiff('x', 2)
        y = 'asd'
        x1 = sad.AutoDiffVector([x, y])

def test_vector_inputlist_duplicate_var():
    with pytest.raises(ValueError):
        x = sad.AutoDiff('x', 2)
        y = sad.AutoDiff('x', 2)
        x1 = sad.AutoDiffVector([x,y])


### 1. Addition

def test_vector_add():
    f1 = sad.AutoDiff('x', 1)
    f2 = sad.AutoDiff('y', 1)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.AutoDiffVector([f1,f2]) + 2
    vec1_other = sad.AutoDiffVector([f1,f2]) + other
    vec2 = sad.vectorize(['x','y'], [1,1]) + 2
    vec2_other = sad.vectorize(['x','y'], [1,1]) + other
    f3 = f1 + 2
    f4 = f1 + other
    #
    der = f3.der['x']
    val = f3.val
    der_other = f4.der['x']
    val_other = f4.val
    do_vector_tests(vec1, vec1_other, vec2, vec2_other, der, der_other, val, val_other)


def test_vector_radd():
    f1 = sad.AutoDiff('x', 1)
    f2 = sad.AutoDiff('y', 1)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = 2 + sad.AutoDiffVector([f1,f2])
    vec1_other = other + sad.AutoDiffVector([f1,f2])
    vec2 = 2 + sad.vectorize(['x','y'], [1,1])
    vec2_other = other + sad.vectorize(['x','y'], [1,1])
    f3 = 2 + f1
    f4 = other + f1
    #
    der = f3.der['x']
    val = f3.val
    der_other = f4.der['x']
    val_other = f4.val
    do_vector_tests(vec1, vec1_other, vec2, vec2_other, der, der_other, val, val_other)

### 2. Subtraction

def test_vector_sub():
    f1 = sad.AutoDiff('x', 1)
    f2 = sad.AutoDiff('y', 1)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.AutoDiffVector([f1,f2]) - 1
    vec1_other = sad.AutoDiffVector([f1,f2]) - other
    vec2 = sad.vectorize(['x','y'], [1,1]) - 1
    vec2_other = sad.vectorize(['x','y'], [1,1]) - other
    f3 = f1 - 1
    f4 = f1 - other
    #
    der = f3.der['x']
    val = f3.val
    der_other = f4.der['x']
    val_other = f4.val
    do_vector_tests(vec1, vec1_other, vec2, vec2_other, der, der_other, val, val_other)

# def test_vector_rsub():
#     f1 = sad.AutoDiff('x', 1)
#     f2 = sad.AutoDiff('y', 1)
#     other = sad.AutoDiff('z',2)
#     # do operations on objects
#     vec1 = 1 - sad.AutoDiffVector([f1,f2])
#     vec1_other = 1 - sad.AutoDiffVector([f1,f2])
#     vec2 = 1 - sad.vectorize(['x','y'], [1,1])
#     vec2_other = other - sad.vectorize(['x','y'], [1,1])
#     f3 = 1 - f1
#     f4 = other - f1
#     #
#     der = f3.der['x']
#     val = f3.val
#     der_other = f4.der['x']
#     val_other = f4.val
#     do_vector_tests(vec1, vec1_other, vec2, vec2_other, der, der_other, val, val_other)


### 3. Multiplication

def test_vector_mul_constant():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.AutoDiffVector([f1,f2]) * 3
    vec1_other = sad.AutoDiffVector([f1,f2]) * other
    vec2 = sad.vectorize(['x','y'], [2,2]) * 3
    vec2_other = sad.vectorize(['x','y'], [2,2]) * other
    f3 = f1 * 3
    f4 = f1 * other
    #
    der = f3.der['x']
    val = f3.val
    der_other = f4.der['x']
    val_other = f4.val
    do_vector_tests(vec1, vec1_other, vec2, vec2_other, der, der_other, val, val_other)

def test_vector_rmul_constant():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = 3 * sad.AutoDiffVector([f1,f2])
    vec1_other = other * sad.AutoDiffVector([f1,f2])
    vec2 = 3 * sad.vectorize(['x','y'], [2,2])
    vec2_other = other * sad.vectorize(['x','y'], [2,2])
    f3 = 3 * f1
    f4 = other * f1
    #
    der = f3.der['x']
    val = f3.val
    der_other = f4.der['x']
    val_other = f4.val
    do_vector_tests(vec1, vec1_other, vec2, vec2_other, der, der_other, val, val_other)


### 4. Additive inverse
def test_vector_neg():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = -sad.AutoDiffVector([f1,f2])
    vec2 = -sad.vectorize(['x','y'], [2,2])
    f3 = -f1
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)


### 5. Division
def test_vector_div():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.AutoDiffVector([f1,f2]) / 3
    vec1_other = sad.AutoDiffVector([f1,f2]) / other
    vec2 = sad.vectorize(['x','y'], [2,2]) / 3
    vec2_other = sad.vectorize(['x','y'], [2,2]) / other
    f3 = f1 / 3
    f4 = f1 / other
    #
    der = f3.der['x']
    val = f3.val
    der_other = f4.der['x']
    val_other = f4.val
    do_vector_tests(vec1, vec1_other, vec2, vec2_other, der, der_other, val, val_other)

# def test_vector_rdiv():
#     f1 = sad.AutoDiff('x', 2)
#     f2 = sad.AutoDiff('y', 2)
#     other = sad.AutoDiff('z',2)
#     # do operations on objects
#     vec1 = 3 / sad.AutoDiffVector([f1,f2])
#     vec1_other = other / sad.AutoDiffVector([f1,f2])
#     vec2 = 3 / sad.vectorize(['x','y'], [2,2])
#     vec2_other = other / sad.vectorize(['x','y'], [2,2])
#     f3 = 3 / f1
#     f4 = other / f1
#     #
#     der = f3.der['x']
#     val = f3.val
#     der_other = f4.der['x']
#     val_other = f4.val
#     do_vector_tests(vec1, vec1_other, vec2, vec2_other, der, der_other, val, val_other)


### 6. Taking powers

def test_vector_pow():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.AutoDiffVector([f1,f2])**3
    vec2 = sad.vectorize(['x','y'], [2,2])**3
    f3 = f1**3
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)

def test_vector_rpow():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = 3.0**sad.AutoDiffVector([f1,f2])
    vec2 = 3.0**sad.vectorize(['x','y'], [2,2])
    f3 = 3.0**f1
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)

### 7. Additional functions from NumPy

def test_vector_sin():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.sin(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.sin(sad.vectorize(['x','y'], [2,2]))
    f3 = sad.sin(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._sinV(x1)

def test_vector_cos():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.cos(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.cos(sad.vectorize(['x','y'], [2,2]))
    f3 = sad.cos(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._cosV(x1)

def test_vector_tan():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.tan(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.tan(sad.vectorize(['x','y'], [2,2]))
    f3 = sad.tan(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._tanV(x1)

def test_vector_arcsin():
    f1 = sad.AutoDiff('x', 0.5)
    f2 = sad.AutoDiff('y', 0.5)
    # do operations on objects
    vec1 = sad.arcsin(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.arcsin(sad.vectorize(['x','y'], [0.5,0.5]))
    f3 = sad.arcsin(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._arcsinV(x1)

def test_vector_arccos():
    f1 = sad.AutoDiff('x', 0.5)
    f2 = sad.AutoDiff('y', 0.5)
    # do operations on objects
    vec1 = sad.arccos(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.arccos(sad.vectorize(['x','y'], [0.5,0.5]))
    f3 = sad.arccos(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._arccosV(x1)

def test_vector_arctan():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    # do operations on objects
    vec1 = sad.arctan(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.arctan(sad.vectorize(['x','y'], [2,2]))
    f3 = sad.arctan(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._arctanV(x1)

def test_vector_exp():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    # do operations on objects
    vec1 = sad.exp(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.exp(sad.vectorize(['x','y'], [2,2]))
    f3 = sad.exp(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._expV(x1)

def test_vector_ln():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.log(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.log(sad.vectorize(['x','y'], [2,2]))
    f3 = sad.log(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._logV(x1)

def test_vector_log10():
    f1 = sad.AutoDiff('x', 2)
    f2 = sad.AutoDiff('y', 2)
    other = sad.AutoDiff('z',2)
    # do operations on objects
    vec1 = sad.log(sad.AutoDiffVector([f1,f2]), base=10)
    vec2 = sad.log(sad.vectorize(['x','y'], [2,2]), base=10)
    f3 = sad.log(f1, base=10)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)

# f(x) = sinh(x); f(0.5) = 0.5210...; f'(x) = cosh(x); f'(0.5) = 1.1276...
def test_vector_sinh():
    f1 = sad.AutoDiff('x', 0.5)
    f2 = sad.AutoDiff('y', 0.5)
    # do operations on objects
    vec1 = sad.sinh(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.sinh(sad.vectorize(['x','y'], [0.5,0.5]))
    f3 = sad.sinh(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._sinhV(x1)

# f(x) = cosh(x); f(0.5) = 1.1276...; f'(x) = sinh(x); f'(0.5) = 0.5210...
def test_vector_cosh():
    f1 = sad.AutoDiff('x', 0.5)
    f2 = sad.AutoDiff('y', 0.5)
    # do operations on objects
    vec1 = sad.cosh(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.cosh(sad.vectorize(['x','y'], [0.5,0.5]))
    f3 = sad.cosh(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._coshV(x1)

# f(x) = tanh(x); f(0.5) = 0.4621...; f'(x) = sech^2(x)= 2/(cos(2x)+1); f'(0.5) = 0.7864...
def test_vector_tanh():
    f1 = sad.AutoDiff('x', 0.5)
    f2 = sad.AutoDiff('y', 0.5)
    # do operations on objects
    vec1 = sad.tanh(sad.AutoDiffVector([f1,f2]))
    vec2 = sad.tanh(sad.vectorize(['x','y'], [0.5,0.5]))
    f3 = sad.tanh(f1)
    #
    der = f3.der['x']
    val = f3.val
    do_vector_tests_no_other(vec1, vec2, der, val)
    with pytest.raises(AttributeError):
        x1 = sad.AutoDiff('x', 1)
        sad._tanhV(x1)





