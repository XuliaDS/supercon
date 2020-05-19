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

def convergence_IO(nR, ne, value, pX, pU, title):
    print("____________________________________________________________________\n")
    print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pU," ----------------")
    print("____________________________________________________________________\n")
    print("\n \t ",title,"\n\n")
    print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
    print("------------------------------------------------------------------------\n")
    for r in range(nR + 1):
        ne1 = pow(2, r) * ne
        if r == 0:
            print (ne1,"\t%1.3e"%value[r],"         |","%1.3e"%numpy.sqrt(value[r]))
        else:
            a = numpy.log10(           value[r-1] /            value[r])  / numpy.log10(2.0)
            b = numpy.log10(numpy.sqrt(value[r-1])/ numpy.sqrt(value[r])) / numpy.log10(2.0)
            print (ne1,"\t%1.3e"%value[r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(value[r]),"  %1.2f" %b)
    print("____________________________________________________________________\n")




def newton_root (guess, eL, eR, polyX, polyT, f):
    tol   = 1.e-15
    itMAX = 20
    r     = 1.0 #0.5 * (eR - eL)
    #eM    = 0.5 * (eR + eL)
    zp    = guess
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


class TestDistanceFunctionOptimization(unittest.TestCase):



    @staticmethod
    def getGeometry1D(c, a, b):
        if   c ==  0 or c == 10: return Curve1DPol.Curve1DCos (a, b)
        elif c ==  1: return Curve1DPol.Curve1DPol1(a, b)
        elif c ==  2: return Curve1DPol.Curve1DPol2(a, b)
        elif c ==  4: return Curve1DPol.Curve1DPol4(a, b)
        elif c ==  5: return Curve1DPol.Curve1DPol5(a, b)
        elif c ==  6: return Curve1DPol.Curve1Dexp (a, b)
        elif c ==  7: return Curve1DPol.Curve1DSine(a, b)
        elif c ==  8: return Curve1DPol.Curve1DCosh(a, b)
        elif c ==  9: return Curve1DPol.Curve1DSinh(a, b)
    @staticmethod
    def getMeshDistances(mesh, parametrization, functionName, tol, gp, fixU = False):

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
            mesh,parametrization,gp)
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

        disparity_e           = numpy.zeros([2,nR + 1])
        disparity_XO          = numpy.zeros([2,nR + 1])
        disparity_XM          = numpy.zeros([2,nR + 1])
        gp                    = 50
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        parametrization       = TestDistanceFunctionOptimization.getGeometry1D(curve, I[0], I[1])


        figcount = 1
        ea       = numpy.zeros(2)

        pltInfo  = '    pX = '+str(pX)+' pT = '+str(pU)
        gpx, uw = quadratures.qType(pX + 1, quadratures.eLGL)
        gpu, pw = quadratures.qType(pU + 1, quadratures.eLGL)
        for ref in range(nR + 1):
            h = (parametrization.theT1 - parametrization.theT0) / (pow (2, ref) * ne)
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

            newMasterElementX = meshO.theMasterElementMakerX.createMasterElement(pX, gp-1)
            newMasterElementU = meshO.theMasterElementMakerU.createMasterElement(pU, gp-1)
            oldMasterElementX = meshI.theMasterElementX
            meshO.theMasterElementX = newMasterElementX
            meshO.theMasterElementU = newMasterElementU

            meshI.theMasterElementX = newMasterElementX
            meshI.theMasterElementU = newMasterElementU

            w   = meshI.theMasterElementX.theGaussWeights
            z   = meshI.theMasterElementX.theGaussPoints
            n   = meshI.theNOfElements
            zx  = numpy.empty((n *  gp     ))
            zp  = numpy.empty((n * (pX + 1)))
            zu  = numpy.empty((n * (pU + 1)))
            eBD = numpy.zeros(n + 1)
            for i in range(n+1):
                 eBD[i] = parametrization.theT0 + h * i

            x     = numpy.empty((2,n * gp))
            t     = numpy.empty((2,n * gp))
            alpha = numpy.empty((2,n * gp))
            xMod  = numpy.empty((2,n * gp))
            errM  = numpy.empty((2,n * gp))
            errT  = numpy.empty((2,n * gp))
            poleC = numpy.empty((2,n * gp))
            poleT = numpy.empty((2,n * gp))
            polyP = numpy.empty((2,n * gp))
            aPF   = numpy.empty((  n * gp))
            dumb  = numpy.zeros([1,1])

            for type in range(2):
                if type == 0: mesh = meshO
                else:         mesh = meshI

                disf,proje,norm    = TestDistanceFunctionOptimization.getMeshDistances(
                                      mesh,parametrization,frechetFunctionName,
                                      tolDistanceCalculation, gp-1)

                disparity_e[type, ref] = disf * disf * 0.5
                        #create interpolating t
                for i in range(n):
                    if type == 0:
                        for j in range(gp):
                            zx[i * gp + j] = 0.5 * ( (eBD[i + 1] - eBD[i]) * z[j] + eBD[i + 1] + eBD[i] )
                            dumb[0]          = zx[i * gp + j]
                            aux              = parametrization.value(dumb)
                            aPF[i * gp + j]  = aux
                            if j < pX + 1:
                                zp[i * (pX + 1) + j]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpx[j] + eBD[i + 1] + eBD[i] )
                            if j < pU + 1:
                                zu[i * (pU + 1) + j]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpu[j] + eBD[i + 1] + eBD[i] )

                    x_i     = mesh.getXElement(i)
                    dx_i    = mesh.getDNXElement(i)
                    t_i     = mesh.getUElement(i)
                    alpha_i = parametrization.value(t_i)

                    # Approximate error function by peaks
                    if (type == 1): pE = max(pX, pU) + 1
                    else:           pE = pX + pU

                    e_leg   = polynomial.polynomial(1, pE, 0)
                    leg_err = e_leg.l2_legPro(x_i - alpha_i, z, w)


                    l2E = polynomial.polynomial(1, pX, 0)
                    l2A = polynomial.polynomial(1, pX, 0)

                    xM_i  = l2A.l2_legPro(x_i, z, w)
                    xA_i  = l2A.l2_legPro(alpha_i, z, w)
                    print(x_i - xM_i)
                    print(' nodes of l2 ')
                    print(l2A.node)
                    print(' modes ',l2A.n)
                    epolP = l2E.l2_legPro(x_i - xA_i, z, w)

                    plt.plot(z, epolP)
                    plt.show()
                    resA  = alpha_i - xA_i

                    resP  = l2E.l2_legPro(resA, z, w)

                    plt.plot(z, resP - epolP)
                    plt.show()
                    plt.plot(z, epolP - resA)
                    plt.show()

                    dxm_i = numpy.zeros(gp)
                    for j in range(gp):
                        aux     = l2A.evaluate(z[j])
                        dxm_i[j]= aux[1,0]

                    sumXO = 0.0
                    sumXM = 0.0
                    #x_i   = xA_i
                    #dx_i  = dxm_i
                    dxa   = x_i  - alpha_i
                    dx2   = xA_i - alpha_i
                    #dx2   = xM_i - x_i# - xM_i

                    for j in range (gp):
                        polyP[type,i * gp + j] = x_i[j] - xA_i[j]
                        x    [type,i * gp + j] = x_i[j]
                        xMod [type,i * gp + j] = xA_i[j]
                        t    [type,i * gp + j] = t_i[j]
                        alpha[type,i * gp + j] = alpha_i[j]
                        poleT[type,i * gp + j] = leg_err[j,0]
                        errT [type,i * gp + j] = dxa[j]
                        errM [type,i * gp + j] = dx2[j]
                        ea[type]               = max (ea[type],dxa[j])
                        sumXO                 += dxa[j] * dxa[j] * w[j] * abs(dx_i[j])
                        sumXM                 += dx2[j] * dx2[j] * w[j] * abs(dx_i[j])

                    disparity_XO  [type,ref] += 0.5 * sumXO
                    disparity_XM  [type,ref] += 0.5 * sumXM

            if ref != 0: continue
            zAXIS  = numpy.zeros(n + 1)
            yAXIS  = numpy.zeros(n + 1)
            ypAXIS = numpy.zeros(n * (pX + 1))
            yuAXIS = numpy.zeros(n * (pU + 1))
            tAXIS  = numpy.zeros([2,n + 1])
            xEP    = numpy.zeros([2,n + 1])
            aEP    = numpy.zeros([2,n + 1])

            for i in range(n):
                zAXIS   [i] = zx      [i * gp]
                tAXIS[0][i] = t    [0][i * gp]
                tAXIS[1][i] = t    [1][i * gp]
                xEP  [0][i] = x    [0][i * gp]
                xEP  [1][i] = x    [1][i * gp]
                aEP  [0][i] = alpha[0][i * gp]
                aEP  [1][i] = alpha[1][i * gp]
            zAXIS[n]    = zx      [-1]
            tAXIS[0][n] = t    [0][-1]
            tAXIS[1][n] = t    [1][-1]
            xEP[0][n]   = x    [0][-1]
            xEP[1][n]   = x    [1][-1]
            aEP[0][n]   = alpha[0][-1]
            aEP[1][n]   = alpha[1][-1]

            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle(' Curves ' + pltInfo)

            for type in range(2):
                plt.subplot(2,3,3 * type + 1)
                if type == 0: plt.title(' Opti     Solution')
                else:         plt.title(' Interpol Solution')
                plt.plot(zx, x[type], c = 'b', linestyle='-.')
                plt.plot(zx,xMod[type], c = 'c', linestyle='--')
                plt.xlabel('z')
                plt.subplot(2,3,3 * type + 2)
                plt.title('Target Solution')
                plt.plot(zx, alpha[type], c = 'r',     linestyle=':', label = 'alpha o t')
                plt.plot(zx, aPF        , c = 'orange', linestyle='-', label = 'alpha')
                plt.xlabel('z')
                plt.legend()
                plt.subplot(2,3,3 * type + 3)
                plt.title(' Overlap')
                plt.plot(zx,     aPF    , c = 'orange', linestyle='-',  label = 'alpha')
                plt.plot(zx,   x[type]  , c = 'b',      linestyle='-.', label = 'x')
                plt.plot(zx,  xMod[type]  , c = 'c',      linestyle='--', label = 'x_modal')
                plt.plot(zx, alpha[type], c = 'r',      linestyle=':',  label = 'alpha o t')
                plt.xlabel('z')
                plt.legend()

            errTN     = numpy.zeros([2, n * gp])
            poleTN    = numpy.zeros([2, n * gp])
            errMN     = numpy.zeros([2, n * gp])
            errTN[0]  =  errT[0] / ea[0]
            errTN[1]  =  errT[1] / ea[1]
            poleTN[0] = poleT[0] / ea[0]
            poleTN[1] = poleT[1] / ea[1]
            errMN[0]  =  errM[0] / ea[0]
            errMN[1]  =  errM[1] / ea[1]
            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle('Error from ' + pltInfo)

            for type in range(2):
                plt.subplot(2,3, 3 * type +1)
                if type == 0: plt.title('Optimized x')
                else:         plt.title('Interpol x')
                plt.plot(zx,  x[type] - aPF, c = 'b', linestyle='-.', label='x - alpha')
                plt.plot(zAXIS,yAXIS       , c = 'g', linewidth = 0.25, linestyle = ':')
                plt.xlabel('z')
                plt.legend()
                plt.subplot(2,3, 3 * type +2)
                if type == 0: plt.title('Optimized x and t ')
                else:         plt.title('Interpol x  and t')
                plt.plot(zx,errT[type], c = 'b', linestyle ='-', label='x - alpha o t')
                plt.plot(zx,poleT[type],c = 'r', linestyle =':', label='error poly')
                plt.plot(zx,errM[type], c = 'c', linestyle ='--', label='x_mod - alpha o t')
                plt.plot(zAXIS,yAXIS,   c = 'g', linewidth = 0.25, linestyle = ':')
                plt.scatter(zAXIS, xEP[type] - aEP[type] , c = 'g', s = 15)
                plt.xlabel('z')
                plt.legend()
                plt.subplot(2,3, 3 * type +3)
                plt.title('Normalized Curves ')
                plt.plot(zx,  errTN[type], c = 'b'  , linestyle='-.', label='x - alpha o t')
                plt.plot(zx,  errMN[type], c = 'c'  , linestyle='--', label='x_mod - alpha o t')
                plt.plot(zx, poleTN[type], c = 'r'  , linestyle=':', label='error poly')
                plt.plot(zAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
                plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)

            fig       = plt.figure(figcount)
            figcount += 1

            plt.suptitle('Error from ' + pltInfo)
            for type in range(2):
                plt.subplot(2,2, 2 * type +1)
                if type == 0: plt.title('Optimized x')
                else:         plt.title('Interpol x')
                plt.plot(zx,   polyP[type], c = 'b', linestyle='-.', label='x - alpha_p')
                plt.plot(zx,   -errM[type], c = 'r', linestyle='-',  label='res_(p+1)')
                plt.plot(   zAXIS,yAXIS,    c = 'g', linewidth = 0.25, linestyle = ':')
                plt.scatter(zAXIS,yAXIS,    c = 'g', s = 15)
                plt.xlabel('z')
                plt.legend()
                plt.subplot(2,2, 2 * type +2)
                if type == 0: plt.title('Optimized x and t ')
                else:         plt.title('Interpol x  and t')
                plt.plot(zx,errT[type], c = 'b', linestyle ='-', label='x - alpha o t')
                plt.plot(zx,poleT[type],c = 'r', linestyle =':', label='error poly')
                plt.plot(zx,polyP[type] + errM[type], c = 'c', linestyle ='--', label='comp error')
                plt.plot(zAXIS,yAXIS,   c = 'g', linewidth = 0.25, linestyle = ':')
                plt.scatter(zAXIS, xEP[type] - aEP[type] , c = 'g', s = 15)
                plt.xlabel('z')
                plt.legend()

        for type in range(2):
            if (type == 0): print(' ********  OPTIMIZED     MESH ***********')
            else:           print(' ********  INTERPOLATING MESH ***********')
            convergence_IO(nR, ne, disparity_e[type]   , pX, pU, 'ELOI       DISPARITY: || x_p (xi) - alpha o t||_sigma')
            convergence_IO(nR, ne, disparity_XO[type]  , pX, pU, 'MY         DISPARITY: || x_p (xi) - alpha o t||_sigma')
            convergence_IO(nR, ne, disparity_XM[type]  , pX, pU, 'MY         DISPARITY: || x_mod (xi) - alpha (xi)||_sigma')
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
    elif curve == 8:  I = [0, 1]
    else: I = [0,1]
    TestDistanceFunctionOptimization.testDistanceFunction(degX, degT, elmts, refine, curve, I, showPlots)
