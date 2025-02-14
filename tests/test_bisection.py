import numpy as np
from assignment1 import bisection as bs
import pytest

def test_check_same_sign():
    test_func = lambda x:x
    a=-1
    b=1
    c=2
    assert not bs.check_same_sign(test_func,a,b)
    assert bs.check_same_sign(test_func,b,c)

def test_bisection_method():
    test_func = lambda x:x
    a=3
    b=-10
    found=bs.bisection_method(test_func,a,b)
    known=0
    assert np.isclose(found,known)

def test_bisection_method():
    test_func = lambda x:x
    a=3
    b=-10
    found=bs.bisection_method(test_func,a,b)
    known=0
    assert np.isclose(found,known)

def test_bisection_method_not_callable_func():
    with pytest.raises(Exception) as e_info:
        bs.bisection_method(1,3,10)

def test_bisection_method_same_sign():
    test_func = lambda x:10
    with pytest.raises(Exception) as e_info:
        bs.bisection_method(test_func,3,10)

def test_bisection_method_max_iter():
    test_func = lambda x:x-10
    a=110
    b=-10
    with pytest.raises(Exception) as e_info:
        bs.bisection_method(test_func,a,b,max_iter=1)