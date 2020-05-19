
import numpy

eGLL = 'Gauss-Lobatto-Legendre'
eGL  = 'Gauss-Legendre'
eGC  = 'Gauss-Chebyshev'
def qType(N, type):

    if type == eGLL:
        N -= 1
        x = numpy.cos(numpy.pi*(numpy.arange(0,N+1))/N)

        P = numpy.zeros(( N+1,N+1) )

        xOld = 2

        while max(abs(x - xOld) > 1.e-10):
            xOld  = x

            P[:,0] = 1
            P[:,1] = x

            for k in range(2,N+1):
                P[:,k] = ((2.0 * float(k) - 1.0 ) * x * P[:,k-1] - float(k-1) * P[:,k-2]) / float(k)

            x = xOld - (x * P[:,N] - P[:,N-1] ) / ( float(N+1) * P[:,N])

        w = 2.0 / ( float(N * (N + 1.0)) * P[:,N]**2)

        x *= -1.0

        return x,w

    elif type == eGL:
        x, w = numpy.polynomial.legendre.leggauss(N)
        #xx = numpy.zeros(N)
        #xx[:] = -x[:]
        return x, w
    else:
        x, w  = numpy.polynomial.chebyshev.chebgauss(N)
        xx = numpy.zeros(N)
        xx[:] = -x[:]
        return xx, w
