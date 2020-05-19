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
            eBD = numpy.zeros(n + 1)
            for i in range(n+1):
                 eBD[i] = parametrization.theT0 + h * i
            x     = numpy.empty((2,n, gp,1))

            err   = numpy.empty((2,n, gp,1))
            polyE = numpy.empty((2,n, gp,1))
            molyE = numpy.empty((2,n, gp,1))
            t     = numpy.empty((2,n, gp,1))
            myt   = numpy.empty((2,n, gp,1))
            dt    = numpy.empty((2,n, gp,1,1))
            eO    = numpy.empty((2,n, gp,1))
            eI    = numpy.empty((2,n, gp,1))
            zx    = numpy.empty((  n, gp,  1))
            zp    = numpy.empty((  n, pX + 1))
            zu    = numpy.empty((  n, pU + 1))
            myalpha = numpy.empty((2,n, gp,1))
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
                        #create interpolating t
                for i in range(n):
                    if type == 0:
                        zx[i]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * z   + eBD[i + 1] + eBD[i] )
                        aPF[i] = parametrization.value(zx[i])
                        zp[i]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpx + eBD[i + 1] + eBD[i] )
                        zu[i]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpu + eBD[i + 1] + eBD[i] )
                    x[type,i]  = mesh.getXElement(i)
                    dx[type,i] = mesh.getDNXElement(i)

                    t[type,i]     = mesh.getUElement(i)
                    alpha[type,i] = parametrization.value(t[type,i])

                    # reconstruct x,t lagrange to take 2nd derivatives
                    idEX    = mesh.theElementsX[i,:]
                    idEU    = mesh.theElementsU[i,:]
                    x_nodes = mesh.theNodes[idEX,:]
                    t_nodes = mesh.theParametricNodes[idEU,:]

                    IY      = numpy.zeros([pX +1,3])
                    IT      = numpy.zeros([pU +1,3])
                    IY[:,0] = x_nodes[:,0]
                    IT[:,0] = t_nodes[:,0]

                    x_lagr  = polynomial.polynomial(pX, 1)
                    x_lagr.interpolate(gpx,IY)
                    t_lagr  = polynomial.polynomial(pU, 1)
                    t_lagr.interpolate(gpu,IT)

                    # Approximate error function by peaks
                    if (type == 1): pE = max(pX, pU) + 1
                    else:           pE = pX + pU

                    zl, wl = quadratures.qType(pE, quadratures.eLGL)

                    IX       = numpy.zeros( pE + 1   )
                    IY       = numpy.zeros([pE + 1,3])
                    IX[   0] = zl[ 0]
                    IX[  -1] = zl[-1]
                    dummy    = numpy.empty((1,1))
                    for k in range(1, pE):
                        IX[k]      = newton_root(0.5 * (zl[k] + zl[k -1]), eBD[i], eBD[i + 1], x_lagr, t_lagr, parametrization)
                    for k in range(pE + 1):
                        xt         = x_lagr.evaluate(-1.0,1.0, IX[k])
                        tt         = t_lagr.evaluate(-1.0,1.0, IX[k])
                        dummy[0,0] = tt[0]
                        IY[k,0]    = xt[0] - float(parametrization.value(dummy))
                    e_lagr = polynomial.polynomial(pE, 1)
                    e_lagr.interpolate(IX,IY)
                    e_leg  = polynomial.polynomial(pE, 0)
                    ef     = x[type,i,:,0] - alpha[type,i,:,0]
                    uvals  = e_leg.l2_legPro(pE, ef , z, w)
                    plt.plot(z,uvals)
                    plt.show()
                    for k in range(gp): print(uvals[k],x[type,i,k])
                    da            = parametrization.tangent(t[type,i])
                    dt[type,i]    = numpy.einsum('li,klj->kij',
                                        mesh.theParametricNodes[mesh.theElementsU[i, :], :],
                                        mesh.theMasterElementU.theShapeFunctionsDerivatives)
                    sumXA    = 0.0
                    sumDXAT  = 0.0

                    for j in range (gp):
                        dadt[type,i,j,0] = da[j,0] * dt[type,i,j,0]
                        aux              = e_lagr.evaluate(z[j,0])
                        polyE[type,i,j]  = aux[0]
                        aux              = e_leg.evaluate(z[j,0])
                        molyE[type,i,j]  = aux[0]
                        err[type,i,j]    = x[type,i,j,0] - alpha[type,i,j,0]
                        dxa              = abs(err[type,i,j]) #* (alpha[type,i,j,0] -  x[type,i,j,0])
                        ddxat            = abs(dx[type,i,j,0] - dadt[type,i,j,0]) #* (dadt[type,i,j,0]  - dx[type,i,j,0])

                        ea[type]         = max ( ea[type], dxa)
                        dea[type]        = max (dea[type], ddxat)
                        ea[type + 2]     = max ( ea[type + 2], abs(x[type,i,j,0] - aPF[i,j,0]))

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
            alphaO    = alpha[0].flatten()
            alphaI    = alpha[1].flatten()
            eO        = err[0].flatten()
            eI        = err[1].flatten()
            ePolO     =  polyE[0].flatten()
            ePolI     =  polyE[1].flatten()
            eMolO     =  molyE[0].flatten()
            eMolI     =  molyE[1].flatten()
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
            #plt.plot(zx,     xO, c = 'c', linestyle='-')
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
        #    plt.plot(zx,     xO, c = 'c', linestyle='--')

            plt.scatter(xAXIS, xiEP, c = 'b', s = 5)
            plt.scatter(xAXIS, aiEP, c = 'r', s = 5)
            plt.xlabel('z')


            errO  = eO / ea[0]
            errI  = eI / ea[1]
            errOO = (xO - aPF) / ea[2]
            errII = (xI - aPF) / ea[3]
            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle('Error from z ' + pltInfo)


            plt.subplot(2,3,1)
            plt.title('Optimized x')
            plt.plot(zx,  xO - aPF, c = 'b'     , linestyle='-.', label='x(z) - alpha(z)')
            plt.scatter(xAXIS, xoEP - aEP, c = 'g', s = 8)
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,3,2)
            plt.title('Optimized x and t')
            plt.plot(zx,eO, c = 'c'     , linestyle='-', label='x(z) - alpha(t)')
            plt.plot(zx,ePolO, c = 'r'  , linestyle=':', label='error poly')
            plt.plot(zx,eMolO, c = 'orange'  , linestyle='-.', label='error modal')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, xoEP - aoEP , c = 'g', s = 15)
            plt.xlabel('z')
            plt.legend()

            ePolO[:] /= ea[0]
            eMolO[:] /= ea[0]
            plt.subplot(2,3,3)
            plt.title('Normalized Curves ')
            plt.plot(zx,      errOO, c = 'b'  , linestyle='-.', label='alpha(z)')
            plt.plot(zx,       errO, c = 'c'  , linestyle='-',   label='alpha(t)')
            plt.plot(zx,ePolO, c = 'r'  , linestyle=':', label='error poly')
            plt.plot(zx,eMolO, c = 'orange'  , linestyle='-.', label='error modal')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, xoEP - aoEP , c = 'g', s = 15)

            plt.subplot(2,3,4)
            plt.title('Interpol x')
            plt.plot(zx,xI - aPF, c = 'b'     , linestyle='-.', label='x(z) - alpha(z)')

            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, xiEP - aEP , c = 'g', s = 15)


            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,3,5)
            plt.title('Interpol x opti t')
            plt.plot(zx,eI, c = 'c'     , linestyle='-', label='x(z) - alpha(t)')
            plt.plot(zx,ePolI, c = 'r'  , linestyle=':', label='error poly')
            plt.plot(zx,eMolI, c = 'orange'  , linestyle='-.', label='error modal')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, xiEP - aiEP , c = 'g', s = 15)

            plt.xlabel('z')
            plt.legend()

            ePolI[:] /= ea[1]
            eMolI[:] /= ea[1]
            plt.subplot(2,3,6)
            plt.title('Normalized Curves ')
            plt.plot(zx,errII, c = 'b'  , linestyle='-.', label='alpha(z)')
            plt.plot(zx,errI, c = 'c'  , linestyle='-',   label='alpha(t)')
            plt.plot(zx,ePolI, c = 'r'  , linestyle=':', label='error poly')
            plt.plot(zx,eMolI, c = 'orange'  , linestyle='-.', label='error modal')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, xiEP - aiEP , c = 'g', s = 15)

            plt.xlabel('z')
            plt.legend()

            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle('Error Function approximation from z ' + pltInfo)


            ePolI[:] *= ea[1]
            eMolI[:] *= ea[1]
            ePolO[:] *= ea[0]
            eMolO[:] *= ea[0]
            plt.subplot(2,2,1)
            plt.title('Optimized x Lagrange ')
            plt.plot(zx,eO - ePolO, c = 'b'  , linestyle='-')
            plt.subplot(2,2,2)
            plt.title('Optimized x Modal ')
            plt.plot(zx,eO - eMolO, c = 'r'  , linestyle=':')
            plt.xlabel('z')

            plt.subplot(2,2,3)
            plt.title('Interpol x Lagrange')
            plt.plot(zx,eI - ePolI, c = 'b'  , linestyle='-')

            plt.subplot(2,2,4)
            plt.title('Interpol x Modal')
            plt.plot(zx,eI - eMolI, c = 'r'  , linestyle=':')
            plt.xlabel('z')
            plt.legend()
            '''
            errO = (dxO - dadtO) / dea[0]
            errI = (dxI - dadtI) / dea[1]
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
            plt.plot(zx,dxO - dadtO, c = 'b'  , linestyle='-')
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
            plt.plot(zx,dxI - dadtI, c = 'b'  , linestyle='-')
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
            '''


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
