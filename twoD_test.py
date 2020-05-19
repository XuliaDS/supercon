#!/bin/python -u
import numpy
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
from Geometry.Curve import CirclePolynomial



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

def euc_dist (a, b, dim):
    c = b - a
    return mynorm(c,dim)

def mynorm (a, dim):
    c = 0.0
    for i in range(dim): c += a[i] * a[i]
    return numpy.sqrt(c)

def orthogonal_projection (x, t, f):
    tol   = 1.e-17
    itMAX = 100
    t_opt = numpy.zeros([len(t), 1])
    tp    = numpy.zeros([1,1])
    tn    = numpy.zeros([1,1])
    for j in range(len(t)):
        tp[0] = t[j,0]
        tn    = tp
        for it in range(itMAX):
            pos  = f.value(tn)
            de   = f.tangent(tn)
            der  = de[0,:,0]
            de   = f.hessian(tn)
            der2 = de[0,:,0]
            vec = x[j] - pos
            fn  = numpy.dot(vec, der)
            dfn = numpy.dot(der, der) - numpy.dot(vec, der2)
            tp  = tn
            if (abs (dfn) < tol):
                print(" NULL DERIVATIVE ", dfn, pos)
                print("p1 ",pos,"  p0", p0)
                return tn
            tn  = tp + fn / dfn
            if (abs(tn - tp) < tol): break
        #print " CONVERGED IN ",it, "ITERATIONS AND ERROR ", tn - tp
        if (it == itMAX): print ("NEWTON didn't converge")
        t_opt[j,0] = tp[0]
    return t_opt


class TestDistanceFunctionOptimization(unittest.TestCase):

    @staticmethod


    def getGeometry2D(c, a, b):
        if c == 0 or c == 10: return Circle.Circle (1.0, a, b) #Curve2DPol.Curve2DCircle(a,b)
        elif c == 1:  return Curve2DPol.Curve2DSine(a, b)
        elif c == 2:  return Curve2DPol.Curve2DPol2(a, b)
        elif c == 3:  return Curve2DPol.Curve2DPol3(a, b)
        elif c == 4:  return Curve2DPol.Curve2DExp (a, b)
        elif c == 5:  return CirclePolynomial.CirclePolynomial(1, 2)
        elif c == 6:  return Curve2DPol.Curve2DPol6(a, b)
        elif c == 7:  return Curve2DPol.Curve2DHypCircle (a, b)
        elif c == 76: return Curve2DPol.Curve2DPol76(a, b)
        elif c == 11: return Curve2DPol.Curve2DSineSine(a, b)
        elif c == -1: return Curve2DPol.Curve2DRoots(a,b)


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
        tolDistanceCalculation = 1.e-10

        tol = 1.e-10

        disparity_e           = numpy.zeros([2,nR + 1])
        disparity_XA          = numpy.zeros([2,nR + 1])
        disparity_DXAT        = numpy.zeros([2,nR + 1])
        gp                    = 50
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        parametrization       =  TestDistanceFunctionOptimization.getGeometry2D(curve, I[0], I[1])
        figcount = 1
        ea       = numpy.zeros(4)
        ea0      = numpy.zeros(4)
        ea1      = numpy.zeros(4)
        dea      = numpy.zeros(2)
        pltInfo  = '    pX = '+str(pX)+' pT = '+str(pU)

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

            newMasterElementX = meshO.theMasterElementMakerX.createMasterElement(pX, gp-1)
            newMasterElementU = meshO.theMasterElementMakerU.createMasterElement(pU, gp-1)

            meshO.theMasterElementX = newMasterElementX
            meshO.theMasterElementU = newMasterElementU

            meshI.theMasterElementX = newMasterElementX
            meshI.theMasterElementU = newMasterElementU

            w   = meshI.theMasterElementX.theGaussWeights
            z   = meshI.theMasterElementX.theGaussPoints
            gpx, uw = quadratures.qType(pX + 1, quadratures.eLGL)
            gpu, pw = quadratures.qType(pU + 1, quadratures.eLGL)
            n   = meshI.theNOfElements
            eBD = numpy.zeros(n + 1)
            for i in range(n+1):
                 eBD[i] = parametrization.theT0 + h * i
            x     = numpy.empty((2,n, gp,2))
            t     = numpy.empty((2,n, gp,1))
            dt    = numpy.empty((2,n, gp,2,1))
            zx    = numpy.empty((  n, gp,    1))
            zp    = numpy.empty((  n, pX + 1))
            zu    = numpy.empty((  n, pU + 1))


            alpha = numpy.empty((2,n, gp,2))
            aPF   = numpy.empty(  (n, gp,2))
            dx    = numpy.empty((2,n, gp,2,1))
            dadt  = numpy.empty((2,n, gp,2))

            for type in range(2):
                if (type == 0): mesh = meshO
                else          : mesh = meshI

                disp,proj,norm = TestDistanceFunctionOptimization.getMeshDistances(
                                        mesh,parametrization,frechetFunctionName,
                                        tolDistanceCalculation, gp)

                disparity_e[type, ref] = disp * disp * 0.5
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

                    for j in range (gp):
                        dadt[type,i,j,0] = da[j,0] * dt[type,i,j,0]
                        dadt[type,i,j,1] = da[j,1] * dt[type,i,j,1]
                        dxa   = euc_dist(alpha[type,i,j] , x[type,i,j], 2)
                        ddxa  = euc_dist( dadt[type,i,j] ,dx[type,i,j], 2)
                        dxfa  = euc_dist(       aPF[i,j]  , x[type,i,j], 2)
                        ea0[type]        = max ( ea[type], abs(alpha[type,i,j,0] - x[type,i,j,0]))
                        ea1[type]        = max ( ea[type], abs(alpha[type,i,j,1] - x[type,i,j,1]))
                        ea[type]         = max ( ea[type], dxa)
                        ea[type + 2]     = max ( ea[type], dxfa)
                        arcLength        = mynorm(dx[type, i,j], 2)
                        derx             = arcLength * w[j]
                        sumXA           +=  dxa  * dxa * derx

                    disparity_XA  [type,ref] += sumXA   * 0.5



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
            xoEP      = numpy.zeros([n + 1,2])
            xiEP      = numpy.zeros([n + 1,2])
            aoEP      = numpy.zeros([n + 1,2])
            aiEP      = numpy.zeros([n + 1,2])
            aEP       = numpy.zeros([n + 1,2])
            for i in range(n):
                xAXIS[i]  = zx[i][0]
                toAXIS[i] = t[0][i][0][0]
                tiAXIS[i] = t[1][i][0][0]
                xoEP[i]   = x[0][i][0]
                xiEP[i]   = x[1][i][0]
                aoEP[i]   = alpha[0][i][0]
                aiEP[i]   = alpha[1][i][0]
                aEP[i]    = aPF[i][0]
            xAXIS[n]  = zx[n-1][-1]
            toAXIS[n] = t[0][n-1][-1]
            tiAXIS[n] = t[1][n-1][-1]
            xoEP[n]   = x[0][n-1][-1]
            xiEP[n]   = x[1][n-1][-1]
            aoEP[n]   = alpha[0][n-1][-1]
            aiEP[n]   = alpha[1][n-1][-1]
            aEP[n]    = aPF[n-1][-1]
            zx        = zx.flatten()
            zp        = zp.flatten()
            zu        = zu.flatten()
            alphaO    = numpy.zeros([n * gp,2])
            alphaI    = numpy.zeros([n * gp,2])
            aPFflt    = numpy.zeros([n * gp,2])
            xO        = numpy.zeros([n * gp,2])
            dxO       = numpy.zeros([n * gp,2])
            dxI       = numpy.zeros([n * gp,2])
            dadtO     = numpy.zeros([n * gp,2])
            dadtI     = numpy.zeros([n * gp,2])
            xI        = numpy.zeros([n * gp,2])
            eucO      = numpy.zeros(n * gp)
            eucI      = numpy.zeros(n * gp)
            for i in range(n):
                for j in range(gp):
                    alphaO[i * gp + j] = alpha[0][i][j]
                    alphaI[i * gp + j] = alpha[1][i][j]
                    aPFflt[i * gp + j] =      aPF[i][j]
                    xO[i * gp + j]     =     x[0][i][j]
                    xI[i * gp + j]     =     x[1][i][j]
                    dxO[i * gp + j]    =    dx[0][i][j][:][0]
                    dxI[i * gp + j]    =    dx[1][i][j][:][0]
                    dadtO[i * gp + j]  =  dadt[0][i][j]
                    dadtI[i * gp + j]  =  dadt[1][i][j]
                    eucO[i * gp + j]   =  euc_dist(xO[i * gp + j], alphaO[i * gp + j],2)
                    eucI[i * gp + j]   =  euc_dist(xI[i * gp + j], alphaI[i * gp + j],2)

            tO        =     t[0].flatten()
            tI        =     t[1].flatten()
            dtO       =    dt[0].flatten()
            dtI       =    dt[1].flatten()


            fig       = plt.figure(figcount)
            figcount += 1

            plt.suptitle(' Curves ' + pltInfo)

            plt.subplot(2,3,1)
            plt.title(' Opti Solution')
            plt.plot(xO[:,0], xO[:,1], c = 'b', linestyle='-.')
            plt.xlabel('x(z)')
            plt.xlabel('y(z)')
            plt.subplot(2,3,2)
            plt.title('Target Curves')
            plt.plot(alphaO[:,0], alphaO[:,1], c = 'r',     linestyle=':', label = 'alpha(t)')
            plt.plot(aPFflt[:,0], aPFflt[:,1], c = 'orange', linestyle='-', label = 'alpha(z)')
            plt.xlabel('x(z)')
            plt.xlabel('y(z)')
            plt.legend()
            plt.subplot(2,3,3)
            plt.title(' Overlap')
            plt.plot(aPFflt[:,0], aPFflt[:,1],  c = 'orange', linestyle='-', label = 'alpha(z)')
            plt.plot(    xO[:,0],     xO[:,1], linestyle='-.', label = 'x(z)')
            plt.plot(alphaO[:,0], alphaO[:,1], c = 'r',     linestyle=':', label = 'alpha(t )')

            plt.scatter(xoEP[:,0], xoEP[:,1], c = 'b', s = 5)
            plt.scatter(aoEP[:,0], aoEP[:,1], c = 'r', s = 5)
            plt.xlabel('x(z)')
            plt.xlabel('y(z)')
            plt.legend()

            plt.subplot(2,3,4)
            plt.title(' Interpol Solution')
            plt.plot(xI[:,0], xI[:,1], c = 'b', linestyle='-.')
            plt.xlabel('x(z)')
            plt.xlabel('y(z)')
            plt.subplot(2,3,5)
            plt.title('Target Curves')
            plt.plot(alphaI[:,0], alphaI[:,1], c = 'r',     linestyle=':', label = 'alpha(t)')
            plt.plot(aPFflt[:,0], aPFflt[:,1], c = 'orange', linestyle='-', label = 'alpha(z)')
            plt.xlabel('x(z)')
            plt.xlabel('y(z)')
            plt.legend()
            plt.subplot(2,3,6)
            plt.title(' Overlap')
            plt.plot(aPFflt[:,0], aPFflt[:,1],  c = 'orange', linestyle='-', label = 'alpha(z)')
            plt.plot(xI[:,0], xI[:,1], linestyle='-.', label = 'x(z)')
            plt.plot(alphaI[:,0], alphaI[:,1], c = 'r',     linestyle=':', label = 'alpha(t )')
            plt.scatter(xiEP[:,0], xiEP[:,1], c = 'b', s = 5)
            plt.scatter(aiEP[:,0], aiEP[:,1], c = 'r', s = 5)
            plt.xlabel('x(z)')
            plt.xlabel('y(z)')
            plt.legend()


            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle('Error  ' + pltInfo)


            plt.subplot(2,4,1)
            plt.title('Optimized x and t (1st comp)')
            plt.plot(zx, alphaO[:,0] - xO[:,0], c = 'c'     , linestyle='-.')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, aoEP[:,0] - xoEP[:,0], c = 'g', s = 15)
            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,4,2)
            plt.title('Optimized x and t (1st comp)')
            plt.plot(zx, alphaO[:,1] - xO[:,1], c = 'b'     , linestyle='--')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, aoEP[:,1] - xoEP[:,1], c = 'g', s = 15)
            plt.xlabel('z')
            plt.legend()


            plt.subplot(2,4,3)
            plt.title('Optimized x and t (distance)')
            plt.plot(   zx, eucO, c = 'r'     , linestyle='-')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,4,4)

            err1 = (alphaO[:,0] - xO[:,0]) / ea0[0]
            err2 = (alphaO[:,1] - xO[:,1]) / ea1[0]
            err3 =               eucO      /  ea[0]

            plt.title('Normalized Curves ')
            plt.plot(zx, err1, c = 'c'  , linestyle='-.', label='1st comp')
            plt.plot(zx, err2, c = 'b'  , linestyle=':' , label='2nd comp')
            plt.plot(zx, err3, c = 'r'  , linestyle='-' , label='euc dist')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 3, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 3, label='z- pT')
            plt.scatter(xAXIS,yAXIS, c = 'g',s = 10)

            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,4,5)
            plt.title('Interpol optimized  t (1st comp)')
            plt.plot(zx, alphaI[:,0] - xI[:,0], c = 'c'     , linestyle='-.')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, aiEP[:,0] - xiEP[:,0], c = 'g', s = 15)
            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,4,6)
            plt.title('Interpol optimized  t (1st comp)')
            plt.plot(zx, alphaI[:,1] - xI[:,1], c = 'b'     , linestyle='--')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.scatter(xAXIS, aiEP[:,1] - xiEP[:,1], c = 'g', s = 15)
            plt.xlabel('z')
            plt.legend()


            plt.subplot(2,4,7)
            plt.title('Interpol optimized  t (distance)')
            plt.plot(   zx, eucI, c = 'r'     , linestyle='-')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 6, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 6, label='z- pT')
            plt.xlabel('z')
            plt.legend()

            plt.subplot(2,4,8)

            err1 = (alphaI[:,0] - xI[:,0]) / ea0[1]
            err2 = (alphaI[:,1] - xI[:,1]) / ea1[1]
            err3 =                    eucI / ea[1]

            plt.title('Normalized Curves ')
            plt.plot(zx, err1, c = 'c'  , linestyle='-.', label='1st comp')
            plt.plot(zx, err2, c = 'b'  , linestyle=':' , label='2nd comp')
            plt.plot(zx, err3, c = 'r'  , linestyle='-' , label='euc dist')
            plt.plot(xAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
            plt.scatter(zp,ypAXIS, c = 'r', s = 3, label='z- pX')
            plt.scatter(zu,yuAXIS, c = 'orange', s = 3, label='z- pT')
            plt.scatter(xAXIS,yAXIS, c = 'g',s = 10)

            plt.xlabel('z')
            plt.legend()


            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle('Approx t --> alpha ^-1  ' + pltInfo)
            plt.subplot(3,1,1)
            plt.plot(zx, tO,color = 'b', linestyle = '-', label='Opti')
            plt.plot(zx, tI, color = 'c', linestyle = ':', label='Interp')
            plt.xlabel('xi')
            plt.legend()
            plt.suptitle('Approx t --> alpha ^-1  ' + pltInfo)
            plt.subplot(3,1,2)
            plt.plot(xO[:,0], tO,color = 'b', linestyle = '-', label='Opti')
            plt.plot(xI[:,0], tI, color = 'c', linestyle = ':', label='Interp')
            plt.xlabel('x_0')
            plt.legend()
            plt.suptitle('Approx t --> alpha ^-1  ' + pltInfo)
            plt.subplot(3,1,3)
            plt.plot(xO[:,1], tO,color = 'b', linestyle = '-', label='Opti')
            plt.plot(xI[:,1], tI, color = 'c', linestyle = ':', label='Interp')
            plt.xlabel('x_1')
            plt.legend()


        for type in range(2):
            if (type == 0): print(' ********  OPTIMIZED     MESH ***********')
            else:           print(' ********  INTERPOLATING MESH ***********')
            #convergence_IO(nR, ne, disparity_e[type]   , pX, pU, 'ELOI       DISPARITY: || x_p (xi) - alpha (xi)||_sigma')
            convergence_IO(nR, ne, disparity_XA[type]  , pX, pU, 'MY         DISPARITY: || x_p (xi) - alpha (xi)||_sigme')
            #convergence_IO(nR, ne, disparity_DXAT[type], pX, pU, '1st-der DISPARITY: || dx_p (xi) - dalpha o dt (xi)||_sigma')
        if (showPlots == True): plt.show()



if __name__ == '__main__':

    argc = len(sys.argv)
    if argc != 7:
        print (" I NEED DEGREEX + degree T + INITIAL ELEMENTS + REFINEMENTS + CURVE TYPE")
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
    elif curve == 76 or curve == -1: I = [0.25, 1.25]

    TestDistanceFunctionOptimization.testDistanceFunction(degX, degT, elmts, refine, curve, I, showPlots)
