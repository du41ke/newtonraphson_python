import numpy as np

def jacobian(x):

    from fun_to_solve import fun as fun

    dx = np.spacing(1)**(1/3) # finite difference step                                                    
    nx = x.size
    f =	fun(x)
    nf = f.size

    J = np.zeros((nf,nx))

    for n in range(1,nx):
        delta = np.zeros((nx,1))
        delta[n] += dx
        dF = fun(x+delta)-fun(x-delta)
        J[:,n] = dF / dx / 2

    return J
