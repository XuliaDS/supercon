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

def legendre (n, z):
    poly = numpy.zeros([gp, n + 1])
    poly[:,0] = 1.0
    for j in range(gp):
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
    poly   = numpy.zeros([gp, n + 1])
    poly[:,0] = 1.0
    for j in range(gp):
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
        disparity_XA          = numpy.zeros([2,nR + 1])
        disparity_DXAT        = numpy.zeros([2,nR + 1])
        gp                    = 50
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        parametrization       = TestDistanceFunctionOptimization.getGeometry1D(curve, I[0], I[1])


        figcount = 1
        ea       = numpy.zeros(4)
        dea      = numpy.zeros(2)
        pltInfo  = '    pX = '+str(pX)+' pT = '+str(pU)
        cro      = pX + pU
        cri      = pU + 1
        gpx, uw = quadratures.qType(pX + 1, quadratures.eLGL)
        gpu, pw = quadratures.qType(pU + 1, quadratures.eLGL)
        gi, uw = quadratures.qType(cri, quadratures.eLGL)
        go, pw = quadratures.qType(cro, quadratures.eLGL)
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

            meshO.theMasterElementX = newMasterElementX
            meshO.theMasterElementU = newMasterElementU

            meshI.theMasterElementX = newMasterElementX
            meshI.theMasterElementU = newMasterElementU

            w   = meshI.theMasterElementX.theGaussWeights
            z   = meshI.theMasterElementX.theGaussPoints
            n   = meshI.theNOfElements
            eBD = numpy.zeros(n + 1)
            for i in range(n+1):
                 eBD[i] = parametrization.theT0 + h * i
            x     = numpy.empty((2,n, gp,1))
            t     = numpy.empty((2,n, gp,1))
            dt    = numpy.empty((2,n, gp,1,1))
            zx    = numpy.empty((  n, gp,  1))
            zp    = numpy.empty((  n, pX + 1))
            zu    = numpy.empty((  n, pU + 1))
            zi    = numpy.empty((  n, cri))
            zo    = numpy.empty((  n, cro))

            alpha = numpy.empty((2,n, gp,1))
            aPF   = numpy.empty((n, gp,1))
            dx    = numpy.empty((2,n, gp,1,1))
            dadt  = numpy.empty((2,n, gp,1))

            for type in range(2):
                if type == 0: mesh = meshO
                else:         mesh = meshI
                disf,proje,norm    = TestDistanceFunctionOptimization.getMeshDistances(
                                      mesh,parametrization,frechetFunctionName,
                                      tolDistanceCalculation, gp-1)

                disparity_e[type, ref] = disf * disf * 0.5
                for i in range(n):
                    x[type,i]     = mesh.getXElement(i)
                    dx[type,i]    = mesh.getDNXElement(i)
                    t[type,i]     = mesh.getUElement(i)
                    alpha[type,i] = parametrization.value(t[type,i])
                    da            = parametrization.tangent(t[type,i])
                    dt[type,i]    = numpy.einsum('li,klj->kij',
                                        mesh.theParametricNodes[mesh.theElementsU[i, :], :],
                                        mesh.theMasterElementU.theShapeFunctionsDerivatives)
                    sumXA    = 0.0
                    sumDXAT  = 0.0

                    if type == 0:
                        zx[i]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * z   + eBD[i + 1] + eBD[i] )
                        aPF[i] = parametrization.value(zx[i])
                        zp[i]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpx + eBD[i + 1] + eBD[i] )
                        zu[i]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpu + eBD[i + 1] + eBD[i] )
                        zi[i]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gi  + eBD[i + 1] + eBD[i] )
                        zo[i]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * go  + eBD[i + 1] + eBD[i] )
                    for j in range (gp):
                        dadt[type,i,j,0] = da[j,0] * dt[type,i,j,0]
                        dxa              = abs(alpha[type,i,j,0] -   x[type,i,j,0]) #* (alpha[type,i,j,0] -  x[type,i,j,0])
                        ddxat            = abs(dadt[type,i,j,0]  -  dx[type,i,j,0]) #* (dadt[type,i,j,0]  - dx[type,i,j,0])

                        ea[type]         = max ( ea[type], dxa)
                        dea[type]        = max (dea[type], ddxat)
                        ea[type + 2]     = max ( ea[type + 2], abs(aPF[i,j,0] - x[type,i,j,0]))

                        derx             = abs(dx[type,i,j]) * w[j]
                        sumXA           +=  dxa  * derx
                        sumDXAT         += ddxat * derx

                    disparity_XA  [type,ref] += sumXA   * 0.5
                    disparity_DXAT[type,ref] += sumDXAT * 0.5



            if ref != 0: continue
            leg_alpha = 'alpha'
            leg_x     = 'x'
            leg_t     = 't(xi)'
            leg_xi    = 'xi(t)'
            leg_comp  = 'alpha(t(xi))'
            xAXIS     = numpy.zeros(n + 1)
            yAXIS     = numpy.zeros(n + 1)
            ypAXIS    = numpy.zeros(n * (pX + 1))
            yuAXIS    = numpy.zeros(n * (pU + 1))
            yiAXIS    = numpy.zeros(n * cri)
            yoAXIS    = numpy.zeros(n * cro)
            toAXIS    = numpy.zeros(n + 1)
            tiAXIS    = numpy.zeros(n + 1)
            xoEP      = numpy.zeros(n + 1)
            xiEP      = numpy.zeros(n + 1)
            aoEP      = numpy.zeros(n + 1)
            aiEP      = numpy.zeros(n + 1)
            aEP       = numpy.zeros(n + 1)
            for i in range(n):
                xAXIS[i]  = zx[i][0]
                toAXIS[i] = t[0][i][0][0]
                tiAXIS[i] = t[1][i][0][0]
                xoEP[i]   = x[0][i][0][0]
                xiEP[i]   = x[1][i][0][0]
                aoEP[i]   = alpha[0][i][0][0]
                aiEP[i]   = alpha[1][i][0][0]
                aEP[i]    = aPF[i][0][0]
            xAXIS[n]  = zx[n-1][-1]
            toAXIS[n] = t[0][n-1][-1][0]
            tiAXIS[n] = t[1][n-1][-1][0]
            xoEP[n]   = x[0][n-1][-1][0]
            xiEP[n]   = x[1][n-1][-1][0]
            aoEP[n]   = alpha[0][n-1][-1][0]
            aiEP[n]   = alpha[1][n-1][-1][0]
            aEP[n]    = aPF[n-1][-1][0]
            zx        = zx.flatten()
            zp        = zp.flatten()
            zu        = zu.flatten()
            zi        = zi.flatten()
            zo        = zo.flatten()
            alphaO    = alpha[0].flatten()
            alphaI    = alpha[1].flatten()
            xO        =     x[0].flatten()
            xI        =     x[1].flatten()
            tO        =     t[0].flatten()
            tI        =     t[1].flatten()
            dtO       =    dt[0].flatten()
            dtI       =    dt[1].flatten()
            dxO       =    dx[0].flatten()
            dxI       =    dx[1].flatten()
            dadtO     =  dadt[0].flatten()
            dadtI     =  dadt[1].flatten()
            aPF       = aPF.flatten()

            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle(' Curves ' + pltInfo)

            plt.subplot(2,3,1)
            plt.title(' Opti Solution')
            plt.plot(zx,     xO, c = 'b', linestyle='-.')
            plt.xlabel('z')
            plt.subplot(2,3,2)
            plt.title('Target s')
            plt.plot(zx, alphaO, c = 'r',     linestyle=':', label = 'alpha(t)')
            plt.plot(zx, aPF   , c = 'orange', linestyle='-', label = 'alpha(z)')
            plt.xlabel('z')
            plt.legend()
            plt.subplot(2,3,3)
            plt.title(' Overlap')
            plt.plot(zx, aPF   , c = 'orange', linestyle='-', label = 'alpha(z)')
            plt.plot(zx,     xO, c = 'b', linestyle='-.', label = 'x(z)')
            plt.plot(zx, alphaO, c = 'r',     linestyle=':', label = 'alpha(t )')

            plt.scatter(xAXIS, xoEP, c = 'b', s = 5)
            plt.scatter(xAXIS, aoEP, c = 'r', s = 5)
            plt.xlabel('z')
            plt.legend()
            plt.subplot(2,3,4)
            plt.title(' Interpol Solution')
            plt.plot(zx,     xI, c = 'b', linestyle='-.')
            plt.plot(zx,     xO, c = 'c', linestyle='-')
            plt.subplot(2,3,5)
            plt.title('Target Curves')
            plt.plot(zx, alphaI, c = 'r',     linestyle=':', label = 'alpha(t )')
            plt.plot(zx, aPF   , c = 'orange', linestyle='-', label = 'alpha(z)')
            plt.legend()
            plt.xlabel('t')
            plt.subplot(2,3,6)
            plt.title(' Overlap')
            plt.plot(zx, aPF   , c = 'orange', linestyle='-', label = 'alpha(z)')
            plt.plot(zx,     xI, c = 'b', linestyle='-.', label = 'x(z)')
            plt.plot(zx, alphaI, c = 'r',     linestyle=':', label = 'alpha(t )')
            plt.plot(zx,     xO, c = 'c', linestyle='--')            

            plt.scatter(xAXIS, xiEP, c = 'b', s = 5)
            plt.scatter(xAXIS, aiEP, c = 'r', s = 5)
            plt.xlabel('z')



            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle('Error from z ' + pltInfo)

            errO  = (alphaO - xO) / ea[0]
            errI  = (alphaI - xI) / ea[1]
            errOO = (aPF    - xO) / ea[2]
            errII = (aPF    - xI) / ea[3]
            plt.subplot(2,3,1)
            plt.title('Optimized x')
            plt.plot(zx,       aPF - xO, c = 'b'     , linestyle='-.', label='x(z) - alpha(z)')
            plt.scatter(xAXIS, aEP - xoEP , c = 'g', s = 8)
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,3,2)
            plt.title('Optimized x and t')
            plt.plot(zx,alphaO - xO, c = 'c'     , linestyle='-', label='x(z) - alpha(t)')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, aoEP - xoEP , c = 'g', s = 15)
            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,3,3)
            plt.title('Normalized Curves ')
            plt.plot(zx,      errOO, c = 'b'  , linestyle='-.', label='alpha(z)')
            plt.plot(zx,       errO, c = 'c'  , linestyle='-',   label='alpha(t)')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, aoEP - xoEP , c = 'g', s = 15)

            plt.subplot(2,3,4)
            plt.title('Interpol x')
            plt.plot(zx,aPF - xI, c = 'b'     , linestyle='-.', label='x(z) - alpha(z)')

            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, aEP - xiEP , c = 'g', s = 15)


            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,3,5)
            plt.title('Interpol x opti t')
            plt.plot(zx,alphaI - xI, c = 'c'     , linestyle='-', label='x(z) - alpha(t)')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, aiEP - xiEP , c = 'g', s = 15)

            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,3,6)
            plt.title('Normalized Curves ')
            plt.plot(zx,errII, c = 'b'  , linestyle='-.', label='alpha(z)')
            plt.plot(zx,errI, c = 'c'  , linestyle='-',   label='alpha(t)')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, aiEP - xiEP , c = 'g', s = 15)

            plt.xlabel('z')
            plt.legend()




            errO = (dadtO - dxO) / dea[0]
            errI = (dadtI - dxI) / dea[1]
            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle(' 1st-derivative Curves ' + pltInfo)
            plt.subplot(2,3,1)
            plt.title(' Opti Solution')
            plt.plot(zx, dadtO, c = 'c', linestyle='-',label='alphaO')
            plt.plot(zx,   dxO, c = 'b', linestyle='-.',label='xO')
            plt.xlabel('xi')
            plt.legend()

            plt.subplot(2,3,2)
            plt.title('Error  ')
            plt.plot(zx,dadtO - dxO, c = 'b'  , linestyle='-')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(xAXIS, yAXIS, c = 'g', s = 15)
            plt.xlabel('z')

            plt.subplot(2,3,3)
            plt.title('Normalized Error ')
            plt.plot(zx,errO, c = 'b'  , linestyle='-')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(xAXIS, yAXIS, c = 'g', s = 15)
            plt.xlabel('z')

            plt.subplot(2,3,4)
            plt.title('Interpolation Solution')
            plt.plot(zx, dadtI, c = 'c', linestyle='-',label='alphaO')
            plt.plot(zx,   dxI, c = 'b', linestyle='-.',label='xO')
            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,3,5)
            plt.title('Error  ')
            plt.plot(zx,dadtI - dxI, c = 'b'  , linestyle='-')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(xAXIS, yAXIS, c = 'g', s = 15)
            plt.xlabel('z')

            plt.subplot(2,3,6)
            plt.title('Normalized Error ')
            plt.plot(zx,errI, c = 'b'  , linestyle='-')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(xAXIS, yAXIS, c = 'g', s = 15)
            plt.xlabel('z')



            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle('Approx t --> alpha ^-1  ' + pltInfo)
            plt.subplot(2,1,1)
            plt.plot(zx, tO,color = 'b', label='Opti')
            plt.plot(zx, tI, color = 'c', label='Interp')
            plt.xlabel('xi')
            plt.legend()

            plt.subplot(2,1,2)
            plt.plot(zx, dtO,color = 'b', label='Opti')
            plt.plot(zx, dtI, color = 'c', label='Interp')
            plt.xlabel('z')
            plt.legend()



        for type in range(2):
            if (type == 0): print(' ********  OPTIMIZED     MESH ***********')
            else:           print(' ********  INTERPOLATING MESH ***********')
            convergence_IO(nR, ne, disparity_e[type]   , pX, pU, 'ELOI       DISPARITY: || x_p (xi) - alpha (xi)||_sigma')
            convergence_IO(nR, ne, disparity_XA[type]  , pX, pU, 'MY         DISPARITY: || x_p (xi) - alpha (xi)||_sigme')
            convergence_IO(nR, ne, disparity_DXAT[type], pX, pU, '1st-der DISPARITY: || dx_p (xi) - dalpha o dt (xi)||_sigma')
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
