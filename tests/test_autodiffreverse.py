import sys
from collections import Counter
import numpy as np
import pytest
import math

sys.path.append('..')
import superautodiff as sad


def test_reverse_add():
	x1 = sad.AutoDiffReverse(4, 'x1')
	x2 = sad.AutoDiffReverse(3, 'x2')
	
	f = x1 + 1
	assert f.der['x1'] ==  pytest.approx(1.0)

	f = 1 + x1
	assert f.der['x1'] ==  pytest.approx(1.0)

	f = x1 + x2
	assert f.der['x1'] ==  pytest.approx(1.0)
	assert f.der['x2'] ==  pytest.approx(1.0)

def test_reverse_sub():
	x1 = sad.AutoDiffReverse(4, 'x1')
	x2 = sad.AutoDiffReverse(3, 'x2')
	
	f = x1 - 1
	assert f.der['x1'] ==  pytest.approx(1.0)

	f = 1 - x1
	assert f.der['x1'] ==  pytest.approx(-1.0)

	f = x1 - x2
	assert f.der['x1'] ==  pytest.approx(1.0)
	assert f.der['x2'] ==  pytest.approx(-1.0)

def test_reverse_mul():
	x1 = sad.AutoDiffReverse(4, 'x1')
	x2 = sad.AutoDiffReverse(3, 'x2')
	
	f = x1 * 2
	assert f.der['x1'] ==  pytest.approx(2.0)

	f = 2 * x1
	assert f.der['x1'] ==  pytest.approx(2.0)

	f = x1 * x2
	assert f.der['x1'] ==  pytest.approx(3.0)
	assert f.der['x2'] ==  pytest.approx(4.0)

def test_reverse_neg():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = -x1
	assert f.der['x1'] ==  pytest.approx(-1.0)

def test_reverse_pow():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = x1 ** 2
	assert f.der['x1'] ==  pytest.approx(8.0)

	f = 2 ** x1
	assert f.der['x1'] ==  pytest.approx(2.0**4*np.log(2))

def test_reverse_div():
	x1 = sad.AutoDiffReverse(4, 'x1')
	x2 = sad.AutoDiffReverse(3, 'x2')
	
	f = x1 / 2
	assert f.der['x1'] ==  pytest.approx(0.5)

	f = 2 / x1
	assert f.der['x1'] ==  pytest.approx(-2.0/(4**2))

	f = x1 / x2
	assert f.der['x1'] ==  pytest.approx(1/3)
	assert f.der['x2'] ==  pytest.approx(-4/(3**2))

def test_reverse_sin():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = sad.sin(x1)
	assert f.der['x1'] ==  pytest.approx(np.cos(4))

def test_reverse_cos():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = sad.cos(x1)
	assert f.der['x1'] ==  pytest.approx(-np.sin(4))

def test_reverse_tan():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = sad.tan(x1)
	assert f.der['x1'] ==  pytest.approx(1 / (np.cos(4) ** 2))

def test_reverse_arcsin():
	x1 = sad.AutoDiffReverse(0.5, 'x1')
	
	f = sad.arcsin(x1)
	assert f.der['x1'] ==  pytest.approx(1 / np.sqrt(1 - 0.5 ** 2))

def test_reverse_arccos():
	x1 = sad.AutoDiffReverse(0.5, 'x1')
	
	f = sad.arccos(x1)
	assert f.der['x1'] ==  pytest.approx(1 / -np.sqrt(1 - 0.5 ** 2))

def test_reverse_arctan():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = sad.arctan(x1)
	assert f.der['x1'] ==  pytest.approx(1 / (1 + 4 * 4))

def test_reverse_exp():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = sad.exp(x1)
	assert f.der['x1'] ==  pytest.approx(np.exp(4))

def test_reverse_log():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = sad.log(x1)
	assert f.der['x1'] ==  pytest.approx(1 / (4 * math.log(math.e)))

def test_reverse_log10():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = sad.log(x1, base=10)
	assert f.der['x1'] ==  pytest.approx(1 / (4 * math.log(10)))

def test_reverse_sqrt():
	x1 = sad.AutoDiffReverse(4, 'x1')
	
	f = sad.sqrt(x1)
	assert f.der['x1'] ==  pytest.approx(1/4)

def test_reverse_backpass():
    x1 = sad.AutoDiffReverse(4, 'x1')
    x2 = sad.AutoDiffReverse(7, 'x2')
    x3 = sad.AutoDiffReverse(3, 'x2')
    f = x1+2*x2-x3*4
    assert f.val == 6
    testdic = reversepass(f.pass_table(),["x1","x2","x3"])
    assert testdic ==  {'x1': 1, 'x2': 2, 'x3': -4}
    f.clear_table()
    








