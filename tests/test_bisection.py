import numpy as np
import bisection as bs

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