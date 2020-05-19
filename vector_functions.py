import numpy


def orientation(x, y):
    det = x[0] * y[1] - y[0] * x[1]      # determinant
    if (det > 0): return 1.0
    return -1.0

def signed_norm(dim, sizex, x, dx):
    if dim == 1:
        if sizex == 1: return x
        else: return x[:,0]
    if sizex == 1:
        ori = orientation(x,dx)
        c   = x[0] * x[0] + x[1] * x[1]
        return ori * numpy.sqrt(c)
    else:
        d = numpy.empty((sizex))
        for i in range(sizex):
            ori  = orientation(x[i],dx[i])
            c    = x[i,0] * x[i,0] + x[i,1] * x[i,1]
            d[i] = ori * numpy.sqrt(c)
    return d

def mydistance (dim, n, x, y):
    return xnorm(dim, n, x - y)

def xnorm (dim, n, x):
    if dim == 1:    return abs(x)
    else:
        if n == 1:
            return numpy.sqrt(x[0] ** 2 + x[1]**2)
        z = numpy.zeros(n)
        for i in range(n):
            z[i] = numpy.sqrt(x[i,0] ** 2 + x[i,1] ** 2)
        return z


def newton_root (guess, eL, eR, polyX, polyT, f):
    tol   = 1.e-15
    itMAX = 20
    r     = 1.0 #0.5 * (eR - eL)
    #eM    = 0.5 * (eR + eL)
    zp   = guess
    zn    = zp
    t     = numpy.zeros([1,1])
    for it in range(itMAX):
        x      = polyX.evaluate(-1.0, 1.0, zn)
        aux    = polyT.evaluate(-1.0, 1.0, zn)
        t[0,0] = aux[0]
        ft     = float(f.tangent(t))
        val    = x[3] - ft * r * aux[3]
        if (abs (val) < tol):
            zp = zn
            break
        dft   = float(f.hessian(t))
        dval  = x[6] - (dft * r * r * aux[3] * aux[3] + ft * aux[6])
        zp    = zn
        if (abs (dval) < tol):
            print(" NULL DERIVATIVE ", dval)
            break
        zn   = zp - val / dval
        if (abs(zn - zp) < tol):
            break
    if (it == itMAX): print ("NEWTON didn't converge")
    return zp
