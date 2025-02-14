import numpy as np
from typing import Union

def check_same_sign(func,u:Union[int,float],v:Union[int,float])->bool:
    '''Given a function and 2 points, check if the signs of the 2 evaluated points are the same.'''
    return np.sign(func(u)) == np.sign(func(v))

def bisection_method(
    func,
    a:Union[int,float],
    b:Union[int,float],
    tol:float=np.finfo(np.float64).eps,
    recursion_count:int=0,
    max_iter:int=1000,
)->Union[int,float]:
    '''
    Given a function and 2 points, find the root of the function within tolerance.
    If bisection method runs more than max_iter iterations, bisection method is not suitable.
    '''

    # if the function has been called recursively max_iter times,
    # bisection method is not suitable to find root.
    if recursion_count > max_iter:
        raise Exception(f'Bisection method cannot find a root within tolerance after {max_iter} iterations.')
    else:
        recursion_count+=1

    # check that func is a callable function
    if not callable(func):
        raise Exception('Please input a callable function!')
    
    # check for the existence of a root
    if check_same_sign(func,a,b):
        raise Exception('Function has no root or multiple roots because sign(func(a)) == sign(func(b)).')

    # bisection method
    c = (a + b)/2
    if np.abs(func(c)) < tol:
        return c
    elif check_same_sign(func,a,c):
        return bisection_method(func,c,b,tol,recursion_count,max_iter)
    elif check_same_sign(func,b,c):
        return bisection_method(func,a,c,tol,recursion_count,max_iter)