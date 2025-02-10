import numpy as np
from typing import Union, List


def newton_1D_approx_next(
    eval_f:Union[float,int],
    eval_fd:Union[float,int],
    x0:Union[float,int]
)->Union[float,int]:
    '''Given the evaluated function, evaluated function derivative, and a guess, provide the next guess.'''
    x1 = x0 - eval_f/eval_fd
    return x1


def newton_1D(
    func:callable,
    f_derivative:callable,
    x0:Union[float,int],
    tol:float=1e-10,
    max_iter:int=100
)->Union[float,int]:
    
    if tol<=0:
        raise ValueError(f'Tolerance of {tol} is too small.')
    if max_iter<1:
        raise ValueError('max_iter must be at least 1.')
    if np.size(x0) > 1:
        raise (f'This function is designed for 1D case only. Initial guess has {np.size(x0)} elements.')

    count=0
    while True:

        if count > max_iter:
            raise(f'Newton method cannot find a solution within {max_iter} iterations.')

        eval_fd = f_derivative(x0)
        eval_f = func(x0)
        x1 = newton_1D_approx_next(eval_f,eval_fd,x0)

        if eval_fd < tol:
            break

        count+=1
        x0=x1

    return x1


def newton_raphson(
    F:callable,
    J:callable,
    x0:List[Union[float,int]],
    tol:float=1e-10,
    max_iter:int=100)->List[Union[float,int]]:
    """
    Given a system of equations, the Jacobian of the system, and an initial guess,
    solve a system of equations using the Newton-Raphson method.

    Parameters:
    F: A function that takes a vector x and returns the value of the system of equations at x.
    J: A function that takes a vector x and returns the Jacobian matrix of the system at x.
    x0: Initial guess for the solution.
    tol: Tolerance for convergence.
    max_iter : Maximum number of iterations.

    Returns:
    x : The approximate solution to the system of equations.
    """

    if tol<=0:
        raise ValueError(f'Tolerance of {tol} is too small.')
    if max_iter<1:
        raise ValueError('max_iter must be at least 1.')
    if np.size(x0) == 1:
            return newton_1D(F,J,x0,tol,max_iter)
    # else:
    #     raise ValueError('Number of elements in initial guess must match number of functions.')
    # need another line to check for Jac and fun

    x = x0
    count_iter=0
    while True:

        if count_iter > max_iter:
            raise(f'Newton method cannot find a solution within {max_iter} iterations.')

        F_x = F(x)
        J_x = J(x)
        
        # Solve the linear system J(x) * delta_x = -F(x)
        delta_x = np.linalg.solve(J_x, -F_x)

        x = x + delta_x

        if np.linalg.norm(delta_x) < tol:
            break
    
    return x