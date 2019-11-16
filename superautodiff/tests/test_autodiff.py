import sys
from collections import Counter
import pytest

sys.path.append('..')
from autodiff import AutoDiff
from functions import *


#### Test TypeErrors

# incorrect number of arguments
def test_input_no_args():
    with pytest.raises(TypeError):
    	x1 = AutoDiff()

def test_input_one_arg():
    with pytest.raises(TypeError):
    	x1 = AutoDiff('x')

def test_input_three_args():
	# TODO:
	pass

def test_input_four_args():
    with pytest.raises(TypeError):
    	x1 = AutoDiff('x',1,2,4)



#### Test ValueErrors (incorrect input types)

# test for list inputs, should raise ValueError for now
# TODO: will be removed for final submission
def test_input_list():
    with pytest.raises(ValueError):
    	x1 = AutoDiff(['x', 'y'], [1, 2])

# differing lengths of inputs
def test_input_list_missing_val():
    with pytest.raises(ValueError):
    	x1 = AutoDiff(['x', 'y'], [1])	

def test_input_list_missing_var():
    with pytest.raises(ValueError):
    	x1 = AutoDiff(['x'], [1,2])

# one dimensional inputs, incorrect type
def test_input_float_var():
    with pytest.raises(ValueError):
    	x1 = AutoDiff(1.0, 2)

def test_input_int_var():
    with pytest.raises(ValueError):
    	x1 = AutoDiff(1, 2)

def test_input_str_val():
    with pytest.raises(ValueError):
    	x1 = AutoDiff('x', '2')

# duplicate var
def test_input_duplicate_var():
    pass
    #TODO: after list inputs implemented
    #with pytest.raises(ValueError):
    #	x1 = AutoDiff(['x','x'], [1,2])



#### Test object attributes

# test if .var is str
def test_var_is_str():
    x1 = AutoDiff('x', 2)
    assert isinstance(x1.var, str)

# test if .val is float
def test_val_is_float():
    x1 = AutoDiff('x', 2)
    assert isinstance(x1.val, float)

# test if .der is dict
def test_der_is_dict():
    x1 = AutoDiff('x', 2)
    #TODO: maybe change to dict?
    assert isinstance(x1.der, Counter)

# test if .der keys are str
def test_der_keys_are_str():
	# TODO: change to incorporate list inputs
    x1 = AutoDiff('x', 2)
    for key in x1.der.keys():
    	assert isinstance(key, str)

# test if .der values are floats
def test_der_values_are_float():
	# TODO: change to incorporate list inputs
    x1 = AutoDiff('x', 2)
    for key in x1.der.keys():
    	value = x1.der[key]
    	assert isinstance(value, float)



#### Test intended behavior, elementary operations
#### All tests that should pass have a commented line above that contains
#### the algebraic solutions for the (hopefully) correct f.der [f'(x)] and f.val [f(x)]

## TODO: Define helper function to check attributes .var, .var, and first key of .der
## to reduce redundancy
# def check_attributes(ad):
#     assert next(iter(ad.der)) == 'x' # checks if first key is 'x'
#     assert ad.var == {'x'} # TODO: change to dict
#     assert ad.val == 2.0

### 0. Correct initialization
# f(x) = x; f(2) = 2; f'(x) = 1; f'(2) = 1
def test_correct_init():
    x1 = AutoDiff('x', 2)
    f = x1
    assert f.der['x'] == 1.0 
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 2.0


### 1. Addition

# f(x) = x + 1; f(2) = 3; f'(x) = 1; f'(2) = 1
def test_add_constant():
    x1 = AutoDiff('x', 2)
    f = x1 + 1.0
    assert f.der['x'] == 1.0 
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 3.0

#REMOVE?
def test_add_str():
    with pytest.raises(TypeError):
	    x1 = AutoDiff('x', 2)
	    f = x1 + '2'

# f(x) = 1 + x; f(2) = 3; f'(x) = 1; f'(2) = 1
def test_radd_constant():
    x1 = AutoDiff('x', 2)
    f = 1.0 + x1
    assert f.der['x'] == 1.0 
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 3.0

# f(x) = x + x; f(2) = 4; f'(x) = 2; f'(2) = 2
def test_add_autodiff_self():
    x1 = AutoDiff('x', 2)
    f = x1 + x1
    assert f.der['x'] == 2.0 
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 4.0

# f(x) = x + x; f(2) = 4; f'(x) = 2; f'(2) = 2
def test_add_autodiff_other():
	x1 = AutoDiff('x', 2)
	x2 = AutoDiff('x', 2)
	f = x1 + x2
	assert f.der['x'] == 2.0 # f(x)=x + x; f'(x) = 2; f'(2) = 2
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == 4.0

def test_add_autodiff_other_conflicting_val():
	with pytest.raises(TypeError):
		x1 = AutoDiff('x', 2)
		x2 = AutoDiff('x', 3)
		f = x1 + x2

# REMOVE? Don't need to check for all different types
def test_add_autodiff_conflicting_val():
	with pytest.raises(TypeError):
		x1 = AutoDiff('x', 2)
		f = x1 + [1]


### 2. Subtraction

# f(x) = x - 1; f(2) = 1; f'(x) = 1; f'(2) = 1
def test_sub_constant():
    x1 = AutoDiff('x', 2)
    f = x1 - 1.0
    assert f.der['x'] == 1.0
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 1.0

def test_sub_str():
    with pytest.raises(TypeError):
	    x1 = AutoDiff('x', 2)
	    f = x1 + '2'

# f(x) = 1 - x; f(2) = 1; f'(x) = 1; f'(2) = 1
def test_rsub_constant():
    x1 = AutoDiff('x', 2)
    f = 1.0 - x1
    assert f.der['x'] == 1.0
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 1.0

# f(x) = x - x; f(2) = 0; f'(x) = 0; f'(2) = 0
def test_sub_autodiff_self():
	x1 = AutoDiff('x', 2)
	f = x1 - x1
	assert f.der['x'] == 0.0
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == 0.0

# f(x) = x - x; f(2) = 0; f'(x) = 0; f'(2) = 0
def test_sub_autodiff_other():
	x1 = AutoDiff('x', 2)
	x2 = AutoDiff('x', 2)
	f = x1 - x2
	assert f.der['x'] == 0.0
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == 0.0

def test_sub_autodiff_other_conflicting_val():
	with pytest.raises(TypeError):
		x1 = AutoDiff('x', 2)
		x2 = AutoDiff('x', 3)
		f = x1 - x2

### 3. Multiplication

# f(x) = 3x; f(2) = 6; f'(x) = 3; f'(2) = 3
def test_mul_constant():
    x1 = AutoDiff('x', 2)
    f = x1 * 3
    assert f.der['x'] == 3.0
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 6.0

def test_mul_str():
    with pytest.raises(TypeError):
	    x1 = AutoDiff('x', 2.0)
	    f = x1 * '3'

# f(x) = 3x; f(2) = 6; f'(x) = 3; f'(2) = 6
def test_rmul_constant():
    x1 = AutoDiff('x', 2)
    f = 3 * x1
    assert f.der['x'] == 3.0
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 6.0

# f(x) = x^2; f(2) = 4; f'(x) = 2x; f'(2) = 4
def test_mul_autodiff_self():
	x1 = AutoDiff('x', 2.0)
	f = x1 * x1
	assert f.der['x'] == 4.0
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == 4.0

# f(x) = x^2; f(2) = 4; f'(x) = 2x; f'(2) = 4
def test_mul_autodiff_other():
	x1 = AutoDiff('x', 2.0)
	x2 = AutoDiff('x', 2.0)
	f = x1 * x2
	assert f.der['x'] == 4.0
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == 4.0

def test_mul_autodiff_other_conflicting_val():
	with pytest.raises(TypeError):
		x1 = AutoDiff('x', 2.0)
		x2 = AutoDiff('x', 3.0)
		f = x1 * x2

### 4. Additive inverse
# f(x) = -x; f(2) = -2; f'(x) = -1; f'(2) = -1
def test_neg():
    x1 = AutoDiff('x', 2)
    f = -x1
    assert f.der['x'] == -1
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == -2


### 5. Division

# f(x) = x/2; f(4) = 2; f'(x) = 1/2; f'(4) = 1/2
def test_div_constant():
    x1 = AutoDiff('x', 4)
    f = x1 / 2
    assert f.der['x'] == 0.5
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 2.0

# f(x) = 2/x; f(4) = 1/2; f'(x) = -2/x^2; f'(4) = -1/8
def test_rdiv_constant():
    x1 = AutoDiff('x', 2)
    f = 2 / x1
    assert f.der['x'] == -0.125
    #Test other attributes
    assert next(iter(f.der)) == 'x' # checks if first key is 'x'
    assert f.var == {'x'} # TODO: change to dict
    assert f.val == 0.5

# f(x) = 1; f(2) = 1; f'(x) = 0; f'(2) = 0
def test_div_autodiff_self():
	x1 = AutoDiff('x', 2.0)
	f = x1 / x1
	assert f.der['x'] == 0.0
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == 1.0

# f(x) = 1; f(2) = 1; f'(x) = 0; f'(2) = 0
def test_div_autodiff_other():
	x1 = AutoDiff('x', 2.0)
	x2 = AutoDiff('x', 2.0)
	f = x1 / x2
	assert f.der['x'] == 0.0
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == 1.0

def test_div_autodiff_other_conflicting_val():
	with pytest.raises(TypeError):
		x1 = AutoDiff('x', 2.0)
		x2 = AutoDiff('x', 3.0)
		f = x1 / x2


### 6. Taking powers

# f(x) = x^2; f(2) = 4; f'(x) = 2x; f'(2) = 4
def test_power():
	x1 = AutoDiff('x', 2.0)
	f = x1**2
	assert f.der['x'] == 4.0
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == 4.0

### 7. Additional functions from NumPy

# f(x) = sin(x); f(2*pi) = 0; f'(x) = cos(x); f'(2pi) = 1
def test_sin():
	x1 = AutoDiff('x', 2*math.pi)
	f = sin(x1)
	assert f.der['x'] == pytest.approx(1)
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(0)

# f(x) = cos(x); f(2*pi) = 1; f'(x) = -cos(x); f'(2pi) = 0
def test_cos():
	x1 = AutoDiff('x', 2*math.pi)
	f = cos(x1)
	assert f.der['x'] == pytest.approx(0)
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(1)

# f(x) = tan(x); f(2*pi) = 0; f'(x) = sec^2(x); f'(2pi) = 1
def test_tan():
	x1 = AutoDiff('x', 2*math.pi)
	f = tan(x1)
	assert f.der['x'] == pytest.approx(1.0)
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(0)

# f(x) = arcsin(x); f(0.5) = 0.5235...; f'(x) = (1-x^2)^(-1/2); f'(0.5) = 1.1547...
def test_arcsin():
	x1 = AutoDiff('x', 0.5)
	f = arcsin(x1)
	#TODO: maybe change rounding to another approach?
	assert f.der['x'] == pytest.approx((1-0.5**2)**(-1/2))
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(np.arcsin(0.5))


# f(x) = arccos(x); f(0.5) = 1.0472...; f'(x) = -(1-x^2)^(-1/2); f'(0.5) = -1.1547...
def test_arccos():
	x1 = AutoDiff('x', 0.5)
	f = arccos(x1)
	#TODO: maybe change rounding to another approach?
	assert f.der['x'] == pytest.approx(-(1-0.5**2)**(-1/2))
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(np.arccos(0.5))


# f(x) = arctan(x); f(0.5) = 0.4636...; f'(x) = (1+x^2)^(-1); f'(0.5) = 4/5
def test_arctan():
	x1 = AutoDiff('x', 0.5)
	f = arctan(x1)
	#TODO: maybe change rounding to another approach?
	assert f.der['x'] == pytest.approx(0.8)
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(np.arctan(0.5))

# f(x) = exp(x); f(2) = e^2; f'(x) = e^x; f'(2) = e^2
def test_exp():
	x1 = AutoDiff('x', 2)
	f = exp(x1)
	assert f.der['x'] == pytest.approx(np.exp(2))
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(np.exp(2))

# f(x) =ln(x); f(2) = ln(2); f'(x) = 1/x; f'(2) = 1/2
def test_ln():
	x1 = AutoDiff('x', 2)
	f = log(x1)
	assert f.der['x'] == pytest.approx(0.5)
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(math.log(2, math.e))

# f(x) =log10(x); f(100) = 10; f'(x) = 1/(x*ln(10)); f'(2) = 1/2
def test_log10():
	x1 = AutoDiff('x', 100)
	f = log(x1, base=10)
	assert f.der['x'] == pytest.approx(1/(100*math.log(10, math.e)))
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(math.log(100, 10))

# f(x) = sinh(x); f(0.5) = 0.5210...; f'(x) = cosh(x); f'(0.5) = 1.1276...
def test_sinh():
	x1 = AutoDiff('x', 0.5)
	f = sinh(x1)
	assert f.der['x'] == pytest.approx(np.cosh(0.5))
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(np.sinh(0.5))

# f(x) = cosh(x); f(0.5) = 1.1276...; f'(x) = sinh(x); f'(0.5) = 0.5210...
def test_cosh():
	x1 = AutoDiff('x', 0.5)
	f = cosh(x1)
	assert f.der['x'] == pytest.approx(np.sinh(0.5))
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(np.cosh(0.5))

# f(x) = tanh(x); f(0.5) = 0.4621...; f'(x) = sech^2(x)= 2/(cos(2x)+1); f'(0.5) = 0.7864...
def test_tanh():
	x1 = AutoDiff('x', 0.5)
	f = tanh(x1)
	assert f.der['x'] == pytest.approx(2/(np.cosh(2*0.5)+1))
	#Test other attributes
	assert next(iter(f.der)) == 'x' # checks if first key is 'x'
	assert f.var == {'x'} # TODO: change to dict
	assert f.val == pytest.approx(np.tanh(0.5))


