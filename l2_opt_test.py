#!/bin/python -u
import numpy
import math
import unittest
import matplotlib.pyplot as plt

import Globals.configPaths

import Optimization.DistanceFunction.DistanceFunctionOptimization
from Optimization.DistanceFunction import OptimizationMaker

import Writers.VTKMeshWriter

import Geometry.FrechetDistance
import Geometry.ProjectionDistance

from Geometry.Curve import Curve
from Geometry.Curve import Curve2DPol
from Geometry.Curve import Curve1DPol
from Geometry.Curve import LogSpiral
from Geometry.Curve import Circle
from Geometry.Curve import Segment

import plotly.graph_objects as go

from Discretization.Meshers import CurveMesher,SurfaceMesher

from Writers.NumpyMeshWriter import NumpyMeshWriter

from Globals.configPython import *

import polynomial
import quadratures

def euc_dist (a, b, dim):
    c = b - a
    return mynorm(c,dim)

def mynorm (a, dim):
    c = 0.0
    for i in range(dim): c += a[i] * a[i]
    return numpy.sqrt(c)

def eval_leg (n, Q, z):
    poly  = numpy.zeros([n + 1])
    der   = numpy.zeros([n + 1])
    value = numpy.zeros(Q)
    tan   = numpy.zeros(Q)
    for j in range(Q):
        poly[0] = 1.0
        if (n > 0):
            poly[1] = z[j]
            der[1]  = 1.0
            if (n > 1):
                for i in range (1, n -1):
                    aux         = (2.0 * i + 1.0) / (i + 1.0)
                    auxx        = float(i) / (i + 1.0)
                    poly[i + 1] = aux * z[j]  *  poly[i]                  - auxx * poly[i - 1]
                    der [i + 1] = aux         * (poly[i] + z[j] * der[i]) - auxx * der [i - 1]
        value[j] = poly[n]
        tan  [j] = der[n]
    return value, tan

def poly_leg (n, Q, u, z):
    poly  = numpy.zeros([n + 1])
    der   = numpy.zeros([n + 1])
    for j in range(Q):
        poly[0] = 1.0
        if (n > 0):
            poly[1] = z[j]
            der[1]  = 1.0
            if (n > 1):
                for i in range (1, n -1):
                    aux         = (2.0 * i + 1.0) / (i + 1.0)
                    auxx        = float(i) / (i + 1.0)
                    poly[i + 1] = aux * z[j]  *  poly[i]                  - auxx * poly[i - 1]
                    der [i + 1] = aux         * (poly[i] + z[j] * der[i]) - auxx * der [i - 1]


    return value, tan
def l2_pro (n, f, z, w):
    leg    = numpy.zeros(n + 1)
    Q      = len(f)
    umodes = numpy.zeros(n + 1)
    uvals  = numpy.zeros([Q, 1])
    poly   = numpy.zeros([len(z), n + 1])
    poly[:,0] = 1.0
    for j in range(len(z)):
        if (n > 0):
            poly[j,1] = z[j]
            if (n > 1):
                for i in range (1, n):
                    poly [j,i + 1] = (2.0 * i + 1.0) / (i + 1.0) * z[j] * poly[j,i] \
                                          - float(i) / (i + 1.0) * poly [j, i - 1]
    for i in range(n + 1):
        for j in range(Q):
            leg[i] += f[j][0] * poly[j,i] * w[j]

    for i in range(n + 1):
        umodes[i] = (2.0 * i + 1.0) * 0.5 * leg[i] # account mass matrix

    for j in range(Q):
        for i in range(n + 1):
            uvals[j][0] += umodes[i] * poly[j,i]
    return umodes, uvals


def orthogonal_projection (x, t, tmin, tmax, f):
    tol   = 1.e-16
    itMAX = 100
    t_opt = numpy.zeros([len(t), 1])
    tp    = numpy.zeros([1,1])
    tn    = numpy.zeros([1,1])
    for j in range(len(t)):
        tp[0] = t[j,0]
        tn    = tp
        for it in range(itMAX):
            pos  = f.value(tn)
            der  = f.tangent(tn)

            der2 =  f.hessian(tn)
            vec  =  x[j] - pos
            fn   =  numpy.dot(vec, der)
            dfn  = -numpy.dot(der, der) - numpy.dot(vec, der2)
            tp   = tn
            #print('it ',it,' pos ', pos, 'target',x[j])
            if (abs (dfn) < tol):
                print(" NULL DERIVATIVE ", dfn, pos)
                break
            tn  = tp - fn / dfn
            if tn > tmax or tn < tmin:
                tn = t[j]
                break
            if (abs(tn - tp) < tol): break
        #print (" CONVERGED IN ",it, "ITERATIONS AND ERROR ", tn - tp)
        if (it == itMAX): print ("NEWTON didn't converge")
        t_opt[j] = tn
    return t_opt

class TestDistanceFunctionOptimization(unittest.TestCase):



    @staticmethod
    def getGeometry1D(c, a, b):
        if   c ==  0 or c == 10: return Curve1DPol.Curve1DCos (a, b)
        elif c ==  1: return Curve1DPol.Curve1DSine(a, b)
        elif c ==  2: return Curve1DPol.Curve1DPol1(a, b)
        elif c ==  3: return Curve1DPol.Curve1DPol2(a, b)
        elif c ==  4: return Curve1DPol.Curve1Dexp (a, b)
        elif c == -1: return Curve1DPol.Curve1Dseg (a, b)

    @staticmethod
    def getMeshDistances(mesh, parametrization, functionName, tol, nOfSubdivisions, fixU = False):

        disparityDistanceComputer=Geometry.FrechetDistance.FrechetDistance(
            mesh,parametrization,
            functionName)

        if fixU:
            oldParametricMask = mesh.theParametricNodesMask.copy()
            mesh.theParametricNodesMask[:] = True

        disparityDistanceComputer.theFTolRel=tol
        disparityDistanceComputer.theXTolRel=tol
        disparityValue,normalError=disparityDistanceComputer.run()

        projectorDistance = Geometry.ProjectionDistance.ProjectionDistance(
            mesh,parametrization,nOfSubdivisions)
        projectorValue = projectorDistance.run()

        if fixU:
            mesh.theParametricNodesMask = oldParametricMask

        return disparityValue, projectorValue, normalError

    @staticmethod
    def testDistanceFunction(pX, pU, ne, nR, curve, I, showPlots):

        gp              = 20
        parametrization = TestDistanceFunctionOptimization.getGeometry1D(curve, I[0], I[1])
        print(parametrization.getName())
        zz,w  = quadratures.qType(gp, quadratures.eLGL)
        z     = numpy.zeros([gp,1])
        for j in range(gp): z[j][0] = zz[j]
        u, du = eval_leg(pX,gp, zz)

        print(u, du)
        for ref in range(nR + 1):
            n = pow (2, ref) * ne
            h = (parametrization.theT1 - parametrization.theT0) / n
            elmts = numpy.zeros(n + 1)
            for i in range(n + 1): elmts[i] = parametrization.theT0 + h * i
            # Iniital Discretization
            for i in range(n):
                zuni = numpy.zeros([gp,1])
                for j in range (gp): zuni[j][0]  = -1.0 + 2.0 * j / (gp -1)
                t_prev     = (elmts[i + 1] - elmts[i]) * zuni * 0.5 + (elmts[i + 1] + elmts[i]) * 0.5
                alpha      = parametrization.value(t_prev)
                plt.figure(3)
                plt.plot(z, alpha, color = 'b', linestyle = '--', label = 'alpha Init')
                plt.plot(zuni, alpha, color = 'r', linestyle = '--', label = 'alpha Init')
                plt.plot(t_prev, alpha, color = 'orange', linestyle = '--', label = 'alpha Init')
                plt.show()
                uM, u_next = l2_pro (pX, alpha, z, w)
                plt.figure(1)
                plt.plot(t_prev, alpha, color = 'b', linestyle = '--', label = 'alpha Init')
                plt.plot(t_prev, u_next, color = 'orange', linestyle = ':', label = 'l2 Init')
                plt.xlabel('t_opt')
                plt.figure(2)
                plt.plot(zuni, t_prev)
                plt.figure(3)
                plt.plot(z, alpha, color = 'b', linestyle = '--', label = 'alpha Init')
                plt.plot(z, u_next, color = 'orange', linestyle = ':', label = 'l2 Init')
                plt.legend()
                plt.xlabel('xi')

            plt.show()
            for i in range(n):
                zint = z + 2.0 * i
                for it in range(1):
                    uM, u_next = l2_pro (pX, alpha, z, w)
                    t_next     = orthogonal_projection(u_next, t_prev, t_prev[0], t_prev[-1], parametrization)
                    plt.plot(zint, t_prev)
                    plt.plot(zint, t_next)
                    plt.show()
                    alpha2       = parametrization.value(t_next)
                    uM2, u_next2 = l2_pro (pX, alpha2, z, w)

                    for j in range(gp):
                        print(' DISTANCE BEFORE ', abs(alpha[j] - u_next[j]), ' NOW ', abs(alpha2[j] - u_next[j]))
                    t_prev     = t_next
                    alpha      = alpha2
                    plt.figure(3)

                    plt.plot(zint, u_next, color = 'r', linestyle = '-',label = 'x it '+str(it))
                    plt.plot(zint, alpha,  color = 'b', linestyle = ':', label = 'alpha it '+str(it))
                    plt.plot(zint, u_next2,color = 'orange', linestyle = '--',label = 'x opt '+str(it))
                    plt.plot(zint, alpha2, color = 'g', linestyle = '-.', label = 'alpha opt '+str(it))

                    plt.figure(8)
                    plt.plot(zint, u_next  - alpha, color = 'r', linestyle = '-',label = 'x it '+str(it))
                    plt.plot(zint, u_next2 - alpha2,color = 'orange', linestyle = '--',label = 'x opt '+str(it))
                    plt.figure(9)
                    plt.plot(t_prev, u_next  - alpha, color = 'b', linestyle = '-',label = 't it '+str(it))
                    plt.plot(t_next, u_next2 - alpha2,color = 'c', linestyle = '--',label = 't opt '+str(it))
                    plt.figure(5)

                    plt.plot(t_prev, u_next, color = 'r', linestyle = '-',label = 'x it '+str(it))
                    plt.plot(t_prev, alpha,  color = 'b', linestyle = ':', label = 'alpha it '+str(it))
                    plt.plot(t_next, u_next2,color = 'orange', linestyle = '--',label = 'x opt '+str(it))
                    plt.plot(t_next, alpha2, color = 'g', linestyle = '-.', label = 'alpha opt '+str(it))
                elmts[i    ] = t_prev[0]
                elmts[i + 1] = t_prev[-1]
            plt.show()

if __name__ == '__main__':

    argc = len(sys.argv)
    if argc != 7:
        print (" I NEED DEGREEX + degree T + INITIAL ELEMENTS + REFINEMENTS + CURVE TYPE + SHOW MESH")
        print(sys.argv)
        quit(1)
    degX   = int(sys.argv[1])  # number of elements
    degT   = int(sys.argv[2])  # number of elements
    elmts  = int(sys.argv[3])  # number of elements
    refine = int(sys.argv[4])  # number of elements
    curve  = int(sys.argv[5])  # number of elements
    showPlots = int(sys.argv[6])  # number of elements
    if   (curve ==  0):
        I = [0, numpy.pi]
        print(" SOLVING COS(x) x in [0, pi]")
    elif (curve == 10):
        I = [0, 2.0 * numpy.pi]
        print(" SOLVING COS(x) x in [0, 2pi]")
    else:             I = [0,1]

    TestDistanceFunctionOptimization.testDistanceFunction(degX, degT, elmts, refine, curve, I, showPlots)
