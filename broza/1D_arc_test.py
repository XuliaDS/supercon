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


def legendre (n, z):
    poly = numpy.zeros([len(z), n + 1])
    poly[:,0] = 1.0
    for j in range(len(z)):
        if (n > 0):
            poly[j,1] = z[j]
            if (n > 1):
                for i in range (1, n):
                    poly [j,i + 1] = (2.0 * i + 1.0) / (i + 1.0) * z[j] * poly[j,i] - float(i) / (i + 1.0) * poly [j, i - 1]
    return poly


def l2_pro (n, f, z, w):
    leg    = numpy.zeros(n + 1)
    Q      = len(f)
    umodes = numpy.zeros(n + 1)
    uvals  = numpy.zeros([Q,1])
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
            leg[i] += f[j] * poly[j,i] * w[j]

    for i in range(n + 1):
        umodes[i] = (2.0 * i + 1.0) * 0.5 * leg[i] # account mass matrix

    for j in range(Q):
        for i in range(n + 1):
            uvals[j,0] += umodes[i] * poly[j,i]
    return umodes, uvals


def orthogonal_projection (x, t, f):
    tol   = 1.e-15
    itMAX = 20
    t_opt = numpy.zeros([len(t), 1])
    tp    = numpy.zeros([1,1])
    tn    = numpy.zeros([1,1])
    for j in range(len(t)):
        tp[0] = t[j,0]
        tn    = tp
        for it in range(itMAX):
            pos  = f.value(tn)
            der  = f.tangent(tn)

            der2 = f.hessian(tn)
            vec = x[j] - pos
            fn  = numpy.dot(vec, der)
            dfn = numpy.dot(der, der) - numpy.dot(vec, der2)
            tp  = tn
            if (abs (dfn) < tol):
                print(" NULL DERIVATIVE ", dfn, pos)
                #print("p1 ",pos,"  p0", p0)
                break
            tn  = tp + fn / dfn
            if (abs(tn - tp) < tol): break
        #print " CONVERGED IN ",it, "ITERATIONS AND ERROR ", tn - tp
        if (it == itMAX): print ("NEWTON didn't converge")
        t_opt[j,0] = tp[0]
    return t_opt

class TestDistanceFunctionOptimization(unittest.TestCase):



    @staticmethod
    def getGeometry1D(c, a, b):
        if   c ==  0 or c == 10: return Curve1DPol.Curve1DCos (a, b)
        elif c ==  1: return Curve1DPol.Curve1DPol1(a, b)
        elif c ==  2: return Curve1DPol.Curve1DPol2(a, b)
        elif c ==  5: return Curve1DPol.Curve1DPol5(a, b)
        elif c ==  6: return Curve1DPol.Curve1Dexp (a, b)
        elif c ==  7: return Curve1DPol.Curve1DSine(a, b)
        elif c ==  8: return Curve1DPol.Curve1DCosh(a, b)
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

        relocateX = False
        fixU      = False
        callFix   = True
        method    = 'Newton'
        tolDistanceCalculation = 1.e-7

        tol = 1.e-7

        disparity_e    = numpy.zeros([2,nR + 1])
        disparity_XA   = numpy.zeros([2,nR + 1])
        disparity_DXAT = numpy.zeros([2,nR + 1])
        nOfSubdivisions       = 30
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        parametrization       = TestDistanceFunctionOptimization.getGeometry1D(curve, I[0], I[1])

        for ref in range(nR + 1):
            n = pow (2, ref) * ne
            h = (parametrization.theT1 - parametrization.theT0) / n

            optimizer = Optimization.DistanceFunction.DistanceFunctionOptimization.DistanceFunctionOptimization(
                parametrization,
                h,pX,pU,
                objectiveFunctionName,
                tol,
                initialP  = pX,
                method    = method,
                relocateX = relocateX,
                fixU      = fixU
                )

            meshO, meshI   = optimizer.run()

            newMasterElementX = meshO.theMasterElementMakerX.createMasterElement(pX, nOfSubdivisions)
            newMasterElementU = meshO.theMasterElementMakerU.createMasterElement(pU, nOfSubdivisions)

            meshO.theMasterElementX = newMasterElementX
            meshO.theMasterElementU = newMasterElementU

            meshI.theMasterElementX = newMasterElementX
            meshI.theMasterElementU = newMasterElementU

            w = meshI.theMasterElementX.theGaussWeights
            z = meshI.theMasterElementX.theGaussPoints

            figcount = 1

            for type in range(2):
                emaxAbs = 0.0
                if type == 0: mesh = meshO
                else:         mesh = meshI
                disf,proje,norm = TestDistanceFunctionOptimization.getMeshDistances(
                                    mesh,parametrization,frechetFunctionName,
                                    tolDistanceCalculation, nOfSubdivisions)

                disparity_e[type, ref] = disf * disf * 0.5
                for i in range(mesh.theNOfElements):
                    x     = mesh.getXElement(i)
                    du    = mesh.getDNXElement(i)
                    t     = mesh.getUElement(i)
                    alpha = parametrization.value(t)
                    for j in range (len(t)):
                        emaxAbs = max (emaxAbs, abs(alpha[j,0] - x[j,0]))

                for i in range(mesh.theNOfElements):

                    x       = mesh.getXElement(i)
                    t       = mesh.getUElement(i)
                    dx      = mesh.getDNXElement(i)

                    alpha    = parametrization.value(t)
                    dalpha   = parametrization.tangent(t)
                    dt       = numpy.einsum('li,klj->kij',
                                 mesh.theParametricNodes[mesh.theElementsU[i, :], :],
                                 mesh.theMasterElementU.theShapeFunctionsDerivatives)
                    dadt     = dt
                    for j in range(len(dalpha)): dadt[j,0] = dalpha[j,0] * dt[j,0]
                    sumXA    = 0.0
                    sumDXAT  = 0.0
                    for j in range(len(t)):
                        dxa   = (alpha[j] -   x[j]) * (alpha[j] -  x[j])
                        ddxat = (dadt[j]  -  dx[j]) * (dadt[j]  - dx[j])
                        derx  = abs(dx[j]) * w[j]

                        sumXA   +=  dxa  * derx
                        sumDXAT += ddxat * derx

                    disparity_XA  [type,ref] += sumXA
                    disparity_DXAT[type,ref] += sumDXAT

                    emax0  = 0.0
                    for j in range (len(t)):
                        emax0 = max (emax0, abs(alpha[j,0] - x[j,0]))
                    '''p_alpha_l2       =  2 * (pX + pU + 1)
                    alphaM, alphaL2  = l2_pro (p_alpha_l2, alpha, z, w)
                    xM, xL2          = l2_pro (pX , x,           z, w)
                    eM, eL2          = l2_pro (p_alpha_l2, alpha - x, z, w)

                    poly_dif_x = numpy.log10(abs(   x - xL2))
                    poly_dif_a = numpy.log10(abs(alpha - alphaL2))
                    ml = numpy.zeros(p_alpha_l2 + 1)
                    for k in range(p_alpha_l2 + 1):
                        ml[k]     = k
                        alphaM[k] = abs(alphaM[k]) #numpy.log10(abs(alphaM[k]))
                        eM[k]     = numpy.log10(abs(eM[k]))'''

                    zx = z + 2.0 * i
                    if ref == nR:
                        if i == mesh.theNOfElements -1:
                            leg_alpha = 'alpha'
                            leg_x     = 'x'
                            leg_t     = 't(xi)'
                            leg_xi    = 'xi(t)'
                            leg_comp  = 'alpha(t(xi))'
                        else:
                            leg_alpha = None
                            leg_x     = None
                            leg_t     = None
                            leg_xi    = None
                            leg_comp  = None
                        if type == 0: title = plt.suptitle('OPTIMIZED MESH')
                        else:         title = plt.suptitle('INTERPOLATIVE MESH')
                        fig       = plt.figure(figcount)
                        figcount += 1
                        title
                        plt.subplot(1,2,1)
                        plt.title(' SOLUTION CURVES   ')
                        plt.plot(zx, alpha[:,0], c = 'c', linestyle='-',label=leg_alpha)
                        plt.plot(zx,   x[:,0],c = 'b', linestyle='-.',label=leg_x)
                        plt.xlabel('xi')
                        if i == mesh.theNOfElements-1: plt.legend()

                        plt.subplot(1,2,2)
                        plt.title(' SOLUTION CURVES   ')
                        plt.plot(t, alpha[:,0], c = 'c', linestyle='-',label=leg_alpha)
                        plt.plot(t,   x[:,0],c = 'b', linestyle='-.',label=leg_x)
                        plt.xlabel('t')
                        if i == mesh.theNOfElements-1: plt.legend()

                        '''
                        plt.suptitle('Solution Curves ')
                        plt.subplot(2,2,1)
                        plt.title(' Original  ')
                        plt.plot(y, alpha[:,0], c = 'c', linestyle='-',label=leg_alpha)
                        plt.plot(y,   x[:,0],c = 'b', linestyle='-.',label=leg_x)
                        plt.xlabel
                        plt.legend()
                        for k in range(2):
                            fig = plt.figure(figcount)
                            figcount += 1
                            if k == 0:
                                plt.xlabel = plt.plt.xlabel('xi')
                                y      = zx
                            else:
                                plt.xlabel = plt.plt.xlabel('t')
                                y      = t[:,0]


                            plt.suptitle('Solution Curves ')
                            plt.subplot(2,2,1)
                            plt.title(' Original  ')
                            plt.plot(y, alpha[:,0], c = 'c', linestyle='-',label=leg_alpha)
                            plt.plot(y,   x[:,0],c = 'b', linestyle='-.',label=leg_x)
                            plt.xlabel
                            plt.legend()

                            plt.subplot(2,2,2)
                            plt.title(' L2 reprojection')
                            plt.plot(y, alphaL2[:,0], c = 'c', linestyle='-',label=leg_alpha)
                            plt.plot(y, xL2[:,0]     ,c = 'b', linestyle='-.',label=leg_x)
                            plt.xlabel
                            plt.legend()

                            plt.subplot(2,2,3)
                            plt.title(' Projection error')
                            plt.plot(y, poly_dif_a, c = 'c', linestyle='-',label=leg_alpha)
                            plt.xlabel
                            plt.legend()

                            plt.subplot(2,2,4)
                            plt.title(' Projection error')
                            plt.plot(y, poly_dif_x ,c = 'b', linestyle='-.',label=leg_x)
                            plt.xlabel
                            plt.legend()

                        '''
                        # ERROR PLOTS
                        fig       = plt.figure(figcount)
                        figcount += 1
                        title
                        ZEROAXIS  = [0,0]
                        plt.subplot(2,1,1)
                        plt.title('Error Curves ')
                        err       = (alpha[:,0] - x[:,0]) / emaxAbs
                        plt.plot(zx,err, c = 'b'  , linestyle='-')
                        INTERVAL  = [zx[0],zx[-1]]
                        plt.scatter(INTERVAL, ZEROAXIS, c = 'r', s = 5)
                        plt.plot(INTERVAL, ZEROAXIS, c = 'red', linewidth = 1, linestyle = ':')
                        plt.xlabel('xi')
                        if i == mesh.theNOfElements-1: plt.legend()

                        plt.subplot(2,1,2)
                        plt.title('Error Curves ')
                        err       = (alpha - x) / emaxAbs
                        plt.plot(t,err, c = 'b'  , linestyle='-')
                        INTERVAL  = [t[0],t[-1]]
                        plt.scatter(INTERVAL, ZEROAXIS, c = 'r', s = 5)
                        plt.plot(INTERVAL, ZEROAXIS, c = 'red', linewidth = 1, linestyle = ':')
                        plt.xlabel('t')
                        if i == mesh.theNOfElements-1: plt.legend()



                        '''fig = plt.figure(figcount)
                        figcount += 1
                        plt.subplot(2,1,1)
                        cm  = plt.cm.get_cmap('jet')
                        plt.scatter(ml, alphaM,  marker = 'v', color = cm((i + 2) / mesh.theNOfElements), s = (i+1) * 5) #1 - i / mesh.theNOfElements)
                        plt.subplot(2,1,2)
                        cm  = plt.cm.get_cmap('jet')
                        plt.scatter(ml, eM,  marker = 'v', color = cm((i + 2) / mesh.theNOfElements), s = (i+1) * 5) #1 - i / mesh.theNOfElements)

                        fig = plt.figure(figcount)
                        figcount += 1
                        cm = plt.cm.get_cmap('jet')
                        '''
                        fig = plt.figure(figcount)
                        figcount += 1
                        title
                        plt.subplot(2,1,1)
                        plt.scatter(zx, t,  label = leg_t, s = 3, color = 'b')
                        plt.xlabel('xi')
                        if i == mesh.theNOfElements-1: plt.legend()

                        plt.subplot(2,1,2)
                        plt.scatter(x, t,label = leg_t, s = 3, color = 'r')
                        plt.xlabel('x')
                        if i == mesh.theNOfElements-1: plt.legend()



                        fig = plt.figure(figcount)
                        figcount += 1
                        title
                        plt.subplot(2,2,1)
                        plt.plot(zx, dx[:,0],    label = leg_x, linestyle = '-', color = 'r')
                        plt.plot(zx, dadt[:,0]   ,label = leg_comp, linestyle = '--', color = 'c')
                        if i == mesh.theNOfElements-1: plt.legend()

                        plt.xlabel('xi')

                        plt.subplot(2,2,2)
                        plt.plot(zx, dx[:,0] - dadt[:,0],   label = leg_comp, linestyle = '--', color = 'c')
                        if i == mesh.theNOfElements-1: plt.legend()

                        plt.xlabel('xi')

                        figcount += 1
                        plt.subplot(2,2,3)
                        plt.plot(t, dx[:,0],    label = leg_x, linestyle = '-', color = 'r')
                        plt.plot(t, dadt[:,0]   ,label = leg_comp, linestyle = '--', color = 'c')
                        if i == mesh.theNOfElements-1: plt.legend()
                        plt.xlabel('t')

                        plt.subplot(2,2,4)
                        plt.plot(t,  dx[:,0] - dadt[:,0],   label = leg_comp, linestyle = '--', color = 'c')
                        plt.xlabel('t')
                        if i == mesh.theNOfElements-1: plt.legend()


        for type in range(2):
            if (type == 0): print(' ********  OPTIMIZED     MESH ***********')
            else:           print(' ********  INTERPOLATING MESH ***********')
            print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pU," ----------------")
            print("\n \t DISPARITY ELOI || x_p^* - alpha t_q || ==>  expect p + q =", pX + pU,"\n\n")
            print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%disparity_e[type,r],"         |","%1.3e"%numpy.sqrt(disparity_e[type,r]))
                else:
                    a = numpy.log10(           disparity_e[type,r-1] /            disparity_e[type,r]) / numpy.log10(2.0)
                    b = numpy.log10(numpy.sqrt(disparity_e[type,r-1])/ numpy.sqrt(disparity_e[type,r])) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%disparity_e[type,r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(disparity_e[type,r]),"  %1.2f" %b)
            print("____________________________________________________________________\n")
            print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pU," ----------------")
            print("____________________________________________________________________\n")
            print("\n \t MY DISPARITY ||x_p^* - alpha t_q || ==>  expect q + 1 =", pU,"\n\n")
            print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%disparity_XA[type,r],"         |","%1.3e"%numpy.sqrt(disparity_XA[type,r]))
                else:
                    a = numpy.log10(           disparity_XA[type,r-1] /            disparity_XA[type,r])  / numpy.log10(2.0)
                    b = numpy.log10(numpy.sqrt(disparity_XA[type,r-1])/ numpy.sqrt(disparity_XA[type,r])) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%disparity_XA[type,r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(disparity_XA[type,r]),"  %1.2f" %b)
            print("____________________________________________________________________\n")
            print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pU," ----------------")
            print("____________________________________________________________________\n")
            print("\n \t MY DISPARITY DERIVATIVE ||dx_p^* - dalpha t_q || ==>  expect q + 1 =", pU,"\n\n")
            print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%disparity_DXAT[type,r],"         |","%1.3e"%numpy.sqrt(disparity_DXAT[type,r]))
                else:
                    a = numpy.log10(           disparity_DXAT[type,r-1] /            disparity_DXAT[type,r])  / numpy.log10(2.0)
                    b = numpy.log10(numpy.sqrt(disparity_DXAT[type,r-1])/ numpy.sqrt(disparity_DXAT[type,r])) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%disparity_DXAT[type,r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(disparity_DXAT[type,r]),"  %1.2f" %b)
            print("____________________________________________________________________\n")
        if (showPlots == True): plt.show()


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
    elif (curve == 5):
        I = [1, 2]
        print(" SOLVING a poly deg 5 ")
    else:  I = [-1, 1]

    TestDistanceFunctionOptimization.testDistanceFunction(degX, degT, elmts, refine, curve, I, showPlots)
