import numpy as np
import newton as nt

def test_newton_1D_approx_next():
    eval_f = 2
    eval_fd = 1
    x0 = 3
    known = 1
    found = nt.newton_1D_approx_next(eval_f,eval_fd,x0)
    assert np.isclose(known,found)

def test_newton_1D():
    func = lambda x: (x-3)**3
    func_deri = lambda x: 3*(x-3)**2
    x0 = 5
    known = 3
    found = nt.newton_raphson(func,func_deri,x0)
    assert np.isclose(known,found)

def test_newton_raphson():
    funcs = lambda x: np.array([x[0]**2-4*x[0]+5-x[1],
                                np.exp(x[0])-x[1]])
    Jac = lambda x: np.array([
        [2*x[0]-4,-1],
        [np.exp(x[0]),-1]
    ])

    x_guess = [0,0]

    found = nt.newton_raphson(funcs,Jac,x_guess)
    known = np.array([0.84630378,2.33101497])
    
    assert np.all(np.isclose(found,known))