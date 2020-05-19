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


def orientation(x, y):
    det = x[0] * y[1] - y[0] * x[1]      # determinant
    if (det > 0): return 1.0
    return -1.0

def mynorm (sizex, x, dx):
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




def convergence_IO(nR, ne, value, pX, pT, title):
    print("____________________________________________________________________\n")
    print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pT," ----------------")
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
    def getGeometry2D(c, a, b):
        if   c == 4: return Curve2DPol.Curve2DExp (a, b)
        elif c == 1: return Curve2DPol.Curve2DSine(a, b)
        elif c == 11:return Curve2DPol.Curve2DSineSine(a, b)
        elif c == 2: return Curve2DPol.Curve2DPol2(a, b)
        elif c == 3: return Curve2DPol.Curve2DPol3(a, b)
        elif c == 6: return Curve2DPol.Curve2DPol6(a, b)

        elif c == 0 or c == 10: return Circle.Circle (1.0, a, b) #Curve2DPol.Curve2DCircle(a,b)
        elif c == 5:  return CirclePolynomial.CirclePolynomial(1, 2)
        elif c == 8:  return Curve2DPol.Curve2DHypCircle(a,b)
        elif c == 9:  return Curve2DPol.Curve2DsinExp(a,b)
        elif c == 11: return Curve2DPol.Curve2D2Exp(a,b)

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
    def testDistanceFunction(pX, pT, ne, nR, curve, I, showPlots):

        relocateX = False
        fixU      = False
        callFix   = True
        method    = 'Newton'
        tolDistanceCalculation = 1.e-10

        tol = 1.e-10

        disparity_e           = numpy.zeros([2,nR + 1])
        disparity_XO          = numpy.zeros([2,nR + 1])
        disparity_XM          = numpy.zeros([2,nR + 1])
        gp                    = 40
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        parametrization       = TestDistanceFunctionOptimization.getGeometry2D(curve, I[0], I[1])


        figcount = 1
        ea       = numpy.zeros(2)

        pltInfo  = '    pX = '+str(pX)+' pT = '+str(pT)
        gpx, uw = quadratures.qType(pX + 1, quadratures.eLGL)
        gpu, pw = quadratures.qType(pT + 1, quadratures.eLGL)
        for ref in range(nR + 1):
            h = (parametrization.theT1 - parametrization.theT0) / (pow (2, ref) * ne)
            optimizer = Optimization.DistanceFunction.DistanceFunctionOptimization.DistanceFunctionOptimization(
                parametrization,
                h,pX,pT,
                objectiveFunctionName,
                tol,
                initialP  = pX,
                method    = method,
                relocateX = relocateX,
                fixU      = fixU
                )

            meshO, meshI   = optimizer.run()

            newMasterElementX = meshO.theMasterElementMakerX.createMasterElement(pX, gp-1)
            newMasterElementU = meshO.theMasterElementMakerU.createMasterElement(pT, gp-1)
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
            zu  = numpy.empty((n * (pT + 1)))
            eBD = numpy.zeros(n + 1)
            for i in range(n+1):
                 eBD[i] = parametrization.theT0 + h * i
            dim   = 2
            x     = numpy.empty((2,n * gp,dim))
            xM    = numpy.empty((2,n * gp,dim))

            t     = numpy.empty((2,n * gp,dim))
            alpha = numpy.empty((2,n * gp,dim))
            exaC  = numpy.empty((2,n * gp,dim))
            exaT  = numpy.empty((2,n * gp))

            emaC = numpy.empty((2,n * gp,dim))
            emaT = numpy.empty((2,n * gp))

            dumb  = numpy.zeros([1,dim])

            polyP = numpy.empty((2,n * gp, dim))

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
                            if j < pX + 1:
                                zp[i * (pX + 1) + j]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpx[j] + eBD[i + 1] + eBD[i] )
                            if j < pT + 1:
                                zu[i * (pT + 1) + j]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpu[j] + eBD[i + 1] + eBD[i] )

                    x_i     = mesh.getXElement(i)
                    dx_i    = mesh.getDNXElement(i)
                    t_i     = mesh.getUElement(i)

                    alpha_i = parametrization.value(t_i)

                    if type >= 0:
                        dum         = numpy.empty((gp, 1))
                        for k in range (gp): dum[k,0] = zx[i * gp + k]

                        alpha_nodes = parametrization.value(dum)
                        pe = max (pT, 2 * pX)
                        pol_t       = polynomial.polynomial(2,     pe    , 0)
                        pol_p       = polynomial.polynomial(2,     pX    , 0)
                        pol_2p      = polynomial.polynomial(2, 2 * pX + 1 , 0)

                        alphat      = pol_t.l2_legPro(alpha_nodes, z, w)
                        xM_i        = pol_p.l2_legPro(alpha_nodes, z, w)

                        legendre    = pol_t.legendre(z, pe + 1)


                        for j in range (pX + 1, 2 * pX  - 1):
                            for k in range(gp):
                                alphat[k] -= pol_t.node[j] * legendre[k,j]
                                #Now project alpha onto a 2p basis
                        alpha2p   = pol_2p.l2_legPro(alphat, z, w)
                        yp        =  pol_p.l2_legPro(alphat, z, w)


                        plt.plot(z, t_i)
                        plt.show()
                        f = open('workfile', 'w+')
                        for k in range(gp):
                            f.write("%f\t" %( z[k]))
                            f.write("%f\n" %(t_i[k]))
                        f.close()
                        quit()
                        plt.subplot(2,1,1)
                        plt.plot(alpha_i[:,0], alpha_i[:,1],linestyle = '-', label = 'alpha o t ')
                        plt.plot( alphat[:,0],  alphat[:,1],linestyle = '-.', label = 'alpha - qmodes ')
                        plt.plot(alpha2p[:,0], alpha2p[:,1],linestyle = '--', label = 'alpha2p')
                        plt.plot(yp[:,0], yp[:,1],linestyle = '--', label = 'alpha_p')
                        plt.plot(x_i[:,0], x_i[:,1],linestyle = '--', label = 'x_p')
                        plt.legend()
                        plt.subplot(2,1,2)
                        plt.plot(zx, alpha_i[:,0] - alphat[:,0],linestyle = '-', label = '1st comp - full ')
                        plt.plot(zx, alpha_i[:,1] - alphat[:,1],linestyle = '-', label = '2nd comp - full ')
                        plt.plot(zx, alpha_i[:,0] - alpha2p[:,0],linestyle = '-.',label = '1st comp - 2p ')
                        plt.plot(zx, alpha_i[:,1] - alpha2p[:,1],linestyle = '-.',label = '2nd comp - 2p ')
                        plt.legend()
                        plt.show()



                        #Now project alpha onto a 2p basis
                        res       = alphat - alpha2p
                        for k in range(gp):
                            alpha_i[k] = alphat[k]
                            x_i[k]     = xM_i[k]
                            aux        = pol_p.evaluate(z[k])
                            dx_i[k,0]  = aux[1,0]
                            dx_i[k,1]  = aux[1,1]

                    sumXO = 0.0

                    dist_xa  = x_i  - alpha_i

                    for j in range (gp):
                        x    [type,i * gp + j] = x_i[j]
                        t    [type,i * gp + j] = t_i[j]
                        alpha[type,i * gp + j] = alpha_i[j]

                        exaC [type,i * gp + j] = dist_xa[j]
                        exaT [type,i * gp + j] = mynorm(1, dist_xa[j], dx_i[j])

                        ea[type]               = max (ea[type], abs(exaT [type,i * gp + j]))
                        norm_der               = numpy.sqrt( dx_i[j,0] * dx_i[j,0]  + dx_i[j,1]  * dx_i[j,1])
                        sumXO                 += exaT [type,i * gp + j] * exaT [type,i * gp + j] * w[j] * norm_der

                    disparity_XO  [type,ref] += 0.5 * sumXO

            if ref != 0: continue
            zAXIS  = numpy.empty((n + 1))
            yAXIS  = numpy.zeros((n + 1))
            ypAXIS = numpy.zeros((n * (pX + 1)))
            yuAXIS = numpy.zeros((n * (pT + 1)))
            tAXIS  = numpy.empty((2,n + 1))
            xEP    = numpy.empty((2,n + 1, dim))
            aEP    = numpy.empty((2,n + 1, dim))

            for i in range(n):
                zAXIS   [i] = zx      [i * gp]
                tAXIS[0][i] = t    [0][i * gp][0]
                tAXIS[1][i] = t    [1][i * gp][0]
                xEP  [0][i] = x    [0][i * gp]
                xEP  [1][i] = x    [1][i * gp]
                aEP  [0][i] = alpha[0][i * gp]
                aEP  [1][i] = alpha[1][i * gp]
            zAXIS[n]    = zx      [-1]
            tAXIS[0][n] = t    [0][-1][0]
            tAXIS[1][n] = t    [1][-1][0]
            xEP[0][n]   = x    [0][-1]
            xEP[1][n]   = x    [1][-1]
            aEP[0][n]   = alpha[0][-1]
            aEP[1][n]   = alpha[1][-1]

            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle(' Curves ' + pltInfo)
            for type in range(2):
                plt.subplot(2,2,2 * type + 1)
                if type == 0: plt.title(' Opti     Solution')
                else:         plt.title(' Interpol Solution')
                plt.plot(   x[type,:,0],    x[type,:,1], c = 'b', linestyle='--', label = 'x_p')
                plt.xlabel('z')

                plt.subplot(2,2,2 * type + 2)
                plt.title(' Overlap')
                plt.plot(   x[type,:,0],     x[type,:,1], c = 'b', linestyle='--', label = 'y_p')
                plt.plot(alpha[type,:,0],alpha[type,:,1], c = 'r', linestyle='--', label = 'alpha o t')
                plt.xlabel('z')
                plt.legend()
            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle(' Error  Curves ' + pltInfo)
            for type in range(2):
                plt.subplot(2,3,3 * type + 1)
                if type == 0: plt.title(' Opti     Solution 1st comp')
                else:         plt.title(' Interpol Solution 1st comp')
                plt.plot(zx, exaC[type,:,0], c = 'b', linestyle='--', label = 'x_p')
                plt.plot(zAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
                plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)
                plt.xlabel('z')
                plt.legend()

                plt.subplot(2,3,3 * type + 2)
                if type == 0: plt.title(' Opti     Solution 2nd comp')
                else:         plt.title(' Interpol Solution 2nd comp')
                plt.plot(zx, exaC[type,:,1], c = 'b', linestyle='--', label = 'x_p')
                plt.plot(zAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
                plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)
                plt.xlabel('z')
                plt.legend()

                plt.subplot(2,3,3 * type + 3)
                if type == 0: plt.title(' Opti     Solution total')
                else:         plt.title(' Interpol Solution total')
                plt.plot(zx, exaT[type], c = 'b', linestyle='--', label = 'x_p')
                plt.plot(zAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
                plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)
                plt.xlabel('z')
                plt.legend()

        for type in range(2):
            if (type == 0): print(' ********  OPTIMIZED     MESH ***********')
            else:           print(' ********  INTERPOLATING MESH ***********')
            convergence_IO(nR, ne, disparity_e[type]   , pX, pT, 'ELOI       DISPARITY: || x_p (xi) - alpha o t||_sigma')
            convergence_IO(nR, ne, disparity_XO[type]  , pX, pT, 'MY         DISPARITY: || x_p (xi) - alpha o t||_sigma')
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
    if ( curve == 0):
        I = [0,numpy.pi]
        print(" SOLVING alpha = (cos(x), sin(x)) x in [0, pi]")
    elif ( curve == 10):
        I = [0,2 * numpy.pi]
        print(" SOLVING alpha = (cos(x), sin(x)) x in [0, 2pi]")
    else: I = [0,1]
    TestDistanceFunctionOptimization.testDistanceFunction(degX, degT, elmts, refine, curve, I, showPlots)
