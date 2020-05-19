import numpy
import quadratures



class polynomial:
    eMod = 'Modal'
    eNod = 'Nodal'
    eChe = 'Chebyshev'
    eLeg = 'Legendre'
    eLag = 'Lagrange'
    eMon = 'Monomial'

    def __init__(self, dim,  degree, type, ptype, z, w, quadrature, f):
        self.basisType = type
        self.dim       = dim
        self.n         = degree + 1   # polynomial order
        self.node      = numpy.zeros([self.n, dim])
        self.T         = numpy.zeros(self.n)
        self.polyType  = ptype
        if self.polyType  == self.eMon: self.basisType = self.eNod
        if self.basisType == self.eMod:
            self.l2_Pro(f, z, w, quadrature)
        else:
            self.interpolate(f, z)
        aux      = self.evaluate(z, len(z))
        self.value    = aux[:,0]
        self.tangent  = aux[:,1]
        self.hessian  = aux[:,2]

    def getType(self):
        return (self.basisType + '-'+ self.polyType)

    def buildBasis(self, order, z, dimZ):
        poly   = numpy.zeros([dimZ, order])
        dpoly  = numpy.zeros([dimZ, order])
        d2poly = numpy.zeros([dimZ, order])
        for j in range(dimZ):
            if dimZ == 1: t = z
            else:         t = z[j]
            if self.basisType == self.eMod:
                if (self.polyType == self.eLeg):
                    poly[j,0] = 1.0
                    if (order > 1):
                        poly [j,1] = t
                        dpoly[j,1] = 1.0
                        for d in range (1, order - 1):
                            aux             = (2.0 * d + 1.0) / (d + 1.0)
                            auxx            =        float(d) / (d + 1.0)
                            poly  [j,d + 1] = aux * t * poly[j,d]                            - auxx * poly  [j,d - 1]
                            dpoly [j,d + 1] = aux     * (       poly[j,d] + t * dpoly [j,d]) - auxx * dpoly [j,d - 1]
                            d2poly[j,d + 1] = aux     * (2.0 * dpoly[j,d] + t * d2poly[j,d]) - auxx * d2poly[j,d - 1]
                else: # Chebyshev !! NOT DEFINED FOR DERIVATIVES !!
                    poly [j,0] = 1.0     # 1st kind: we will use those
                    if (order > 1):
                        poly [j,1] =       t
                        #poly [j,1] = 2.0 * t
                        for d in range (1, order - 1):
                            poly [j,d + 1] = 2.0 * t * poly [j,d] - poly [j, d - 1]

            elif self.basisType == self.eLag:
                for d in range (order):
                    poly  [j,d] = self.basis  (t,d)
                    dpoly [j,d] = self.basisD (t,d)
                    d2poly[j,d] = self.basisD2(t,d)
            else:
                for d in range (order):
                    poly[j,d]   = pow(t, d)
                    if d == 0: continue
                    dpoly[j,d]  = pow(t, j - 1)
                    if j == 1: continue
                    d2poly[j,d] = pow(t, j - 2)
        return poly, dpoly, d2poly

    def getBasis(self, order, z, dimZ):
        basis, der_basis, der2_basis = self.buildBasis(order, z, len(z))
        return basis


    def evaluate(self, z, dimZ):
        basis, der_basis, der2_basis = self.buildBasis(self.n, z, len(z))
        res                          = numpy.zeros([dimZ, 3, self.dim]) # value, derivative, 2nd derivative,
        for j in range(dimZ):
            for i in range(self.n):
                for k in range(self.dim):
                    res[j,0,k] += self.node[i][k] *      basis[j,i]
                    res[j,1,k] += self.node[i][k] *  der_basis[j,i]
                    res[j,2,k] += self.node[i][k] * der2_basis[j,i]
        return res

    def interpolate(self, X, T):
        if (self.polyType == self.eMon):
            # Vandermonde matrix
            n = len(T)
            V = numpy.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    if (j == 0): V[i,j] = 1.0
                    else:        V[i,j] = pow(T[i], j)
            f = numpy.zeros(n)
            if (self.dim == 1):
                a = numpy.linalg.solve(V, X)
                for i in range(n): self.node[i,0] = a[i]
            else:
                for d in range (self.dim):
                    for i in range(n): f[i] = X[i,d]
                    a = numpy.linalg.solve(V, f)
                    for i in range(n): self.node[i,d] = a[i]
        else:
            self.T    = T
            self.node = X



    def getPolyWeights(self, n):
        w = numpy.zeros(n)
        if self.polyType == self.eLeg:
            for i in range(self.n): w[i] = 2.0 / (2.0 * i + 1.0)
        elif self.polyType == self.eChe:
            w[:] = numpy.pi * 0.5
            w[0] = numpy.pi
        else:
            print(' cant set weihght for ', self.polyType)
            quit()
        return w

    def l2_Pro (self, f, z, w, qtype):
        phi   = numpy.zeros([self.n, self.dim])
        Q     = len(z)
        uvals = numpy.zeros([Q, self.dim])
        poly, dpoly, d2poly = self.buildBasis(self.n, z, len(z))
        polyWeights         = self.getPolyWeights(self.n)
        for i in range(self.n):
            basis_sum = 0.0
            cheat = 0.0
            for j in range(Q):
                wf = 1.0
                if qtype == quadratures.eGC and self.polyType == self.eLeg:
                    wf = numpy.sqrt(1.0 - z[j] * z[j])
                elif qtype != quadratures.eGC and self.polyType == self.eChe:
                    wf = 1.0 / numpy.sqrt(1.0 - z[j] * z[j])
                phi[i] += f[j] * poly[j,i] * wf * w[j]
                basis_sum += wf * w[j]
                cheat += -0.5 * numpy.pi * w[j]
        for i in range(self.n):
            for k in range(self.dim):
                self.node[i,k] = phi[i,k] / polyWeights[i] # account for polynomial inner product

        for j in range(Q):
            for i in range(self.n):
                for k in range(self.dim):
                    uvals[j,k] += self.node[i,k] * poly[j,i]
        return uvals


    def basis(self, x, j):
        b = [(x - self.T[m]) / (self.T[j] - self.T[m])
             for m in range(self.n) if m != j]
        return numpy.prod(b, axis=0)

    def basis1(self, x, j, k):
        b = [(x - self.T[m]) / (self.T[j] - self.T[m])
             for m in range(self.n) if m != j and m != k]
        if len(b) == 0: return 1.0
        return numpy.prod(b, axis=0)
    def basis2(self, x, j, k, i):
        b = [(x - self.T[m]) / (self.T[j] - self.T[m])
             for m in range(self.n) if m != j and m != k and m != i]
        if len(b) == 0: return 1.0
        return numpy.prod(b, axis=0)

    def basisD(self, x, j):
        b = [(self.basis1(x, j, m) / (self.T[j] - self.T[m]))
             for m in range(self.n) if m != j ]
        return numpy.sum(b, axis=0)

    def basisD2(self, x, j):
        sum = 0.0
        for i in range(self.n):
            if j == i: continue
            b = [(self.basis2(x, j, i, m) / (self.T[j] - self.T[m]))
                for m in range(self.n) if m != i and m != j]
            sum += numpy.sum(b, axis=0) / (self.T[j] - self.T[i])
        return sum
