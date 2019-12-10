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











