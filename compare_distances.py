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

from Geometry.Curve import Curve2DPol
from Geometry.Curve import LogSpiral
from Geometry.Curve import Circle


from Discretization.Meshers import CurveMesher,SurfaceMesher

from Writers.NumpyMeshWriter import NumpyMeshWriter

from Globals.configPython import *

import polynomial
import quadratures

def ortho_project (p0, xL, xR, t0, f, type):
    vec = numpy.zeros(3)
    tol   = 1.e-15
    itMAX = 100

    tp    = t0
    tn    = tp

    for it in range(itMAX):

        if (type == 2):
            taux = tn * (xR - xL) * 0.5 + (xR + xL) * 0.5
            pos  = f(taux)
        else: pos = f.evaluate(xL, xR, tn)

        vec[0] = p0[0] - pos[0]
        vec[1] = p0[1] - pos[1]
        vec[2] = p0[2] - pos[2]

        fn  =    vec[0] * pos[3] + vec[1] * pos[4] + vec[2] * pos[5]
        a   =    vec[0] * pos[6] + vec[1] * pos[7] + vec[2] * pos[8]
        b   =    pos[3] * pos[3] + pos[4] * pos[4] + pos[5] * pos[5]
        dfn = b - a
        tp  = tn
        if (abs (dfn) < tol):
            print(" NULL DERIVATIVE ", dfn, pos)
            print("p1 ",pos,"  p0", p0)
            return tn
        tn  = tp + fn / dfn
        if (abs(tn - tp) < tol): break
    #print " CONVERGED IN ",it, "ITERATIONS AND ERROR ", tn - tp
    if (it == itMAX): print ("NEWTON didn't converge")
    return tn


eModal = 0
eLagr  = 1


def chord (p1, p2, deg):
    a = (p2[0] - p1[0]) * (p2[0] - p1[0])
    b = (p2[1] - p1[1]) * (p2[1] - p1[1])
    c = (p2[2] - p1[2]) * (p2[2] - p1[2])
    return pow (a + b + c, deg)

def mag_df(t):
    der = df (t)
    return pow (der[0] * der[0] + der[1] * der[1] + der[2] * der[2], 0.5)

class TestDistanceFunctionOptimization(unittest.TestCase):

    @staticmethod
    def getGeometry2D(c, a, b):
        if   c == 1: return Curve2DPol.Curve2DPol2(a, b)
        elif c == 2: return Curve2DPol.Curve2DPol3(a, b)
        elif c == 3: return Curve2DPol.Curve2DPol4(a, b)
        elif c == 4: return LogSpiral.LogSpiral(1, 0.1, a , b)
        elif c == 5: return Curve2DPol.Curve2DCircle(a, b)
        elif c == 6: return Curve2DPol.Curve2DPol6(a, b)
        elif c == 7: return Curve2DPol.Curve2DSine(a, b)

    @staticmethod
    def getMeshDistances(mesh, parameterization, functionName, tol, nOfSubdivisions, fixU = False):

        disparityDistanceComputer=Geometry.FrechetDistance.FrechetDistance(
            mesh,parameterization,
            functionName)

        if fixU:
            oldParametricMask = mesh.theParametricNodesMask.copy()
            mesh.theParametricNodesMask[:] = True

        disparityDistanceComputer.theFTolRel=tol
        disparityDistanceComputer.theXTolRel=tol
        disparityValue,normalError=disparityDistanceComputer.run()

        projectorDistance = Geometry.ProjectionDistance.ProjectionDistance(
            mesh,parameterization,nOfSubdivisions)
        projectorValue = projectorDistance.run()

        if fixU:
            mesh.theParametricNodesMask = oldParametricMask

        return disparityValue, projectorValue, normalError

    @staticmethod
    def testDistanceFunction(pX, ne, nR, curve, f, I):

        relocateX      = False
        fixU           = False
        callFix        = True
        method         = 'Newton'

        tolDistanceCalculation = 1.e-7

        tol = 1.e-7
        pU  = 4 * pX

        disparity0  = numpy.zeros(nR + 1)
        disparity1  = numpy.zeros(nR + 1)
        disparityF0 = numpy.zeros(nR + 1)
        disparityF1 = numpy.zeros(nR + 1)

        l2          = numpy.zeros([5, nR + 1])
        li          = numpy.zeros([5, nR + 1])

        nOfSubdivisions       = 25
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"

        objectiveFunctionName2 = "Galerkin" #"Intrinsic"
        frechetFunctionName2   = "Galerkin" #"Intrinsic"

        for ref in range(nR + 1):
            n = pow (2, ref) * ne
            h = (I[1] - I[0]) / n

            parameterization = TestDistanceFunctionOptimization.getGeometry2D(curve, I[0], I[1])

            optimizer = Optimization.DistanceFunction.DistanceFunctionOptimization.DistanceFunctionOptimization(
                parameterization,
                h,pX,pU,
                objectiveFunctionName,
                tol,
                initialP  = pX,
                method    = method,
                relocateX = relocateX,
                fixU      = fixU
                )

            mesh,initialMesh = optimizer.run()


            disparityDistance0,projectionDistance0,normalError0 = \
                TestDistanceFunctionOptimization.getMeshDistances(
                    initialMesh,parameterization,frechetFunctionName,
                    tolDistanceCalculation, nOfSubdivisions)

            disparityDistance1,projectionDistance1,normalError1 = \
                TestDistanceFunctionOptimization.getMeshDistances(
                    mesh,parameterization,frechetFunctionName,
                    tolDistanceCalculation, nOfSubdivisions)

            disparity0 [ref] = disparityDistance0  * disparityDistance0  * 0.5
            disparity1 [ref] = disparityDistance1  * disparityDistance1  * 0.5

            if callFix == True:
                parameterization2 = TestDistanceFunctionOptimization.getGeometry2D(curve, I[0], I[1])

                optimizerFix = Optimization.DistanceFunction.DistanceFunctionOptimization.DistanceFunctionOptimization(
                    parameterization2,
                    h,pX,pU,
                    objectiveFunctionName2,
                    tol,
                    initialP  = pX,
                    method    = method,
                    relocateX = relocateX,
                    fixU      = fixU
                    )

                meshFix,initFix = optimizerFix.run()

                disparityDistanceF0,projectionDistanceF0,normalErrorF0 = \
                    TestDistanceFunctionOptimization.getMeshDistances(
                        initFix,parameterization2,frechetFunctionName2,
                        tolDistanceCalculation, nOfSubdivisions)

                disparityDistanceF1,projectionDistanceF1,normalErrorF1 = \
                    TestDistanceFunctionOptimization.getMeshDistances(
                        meshFix,parameterization2,frechetFunctionName2,
                        tolDistanceCalculation, nOfSubdivisions)
                disparityF0[ref] = disparityDistanceF0 * disparityDistanceF0 * 0.5
                disparityF1[ref] = disparityDistanceF1 * disparityDistanceF1 * 0.5



            w         = mesh.theMasterElementX.theGaussWeights
            z         = mesh.theMasterElementX.theGaussPoints

            lagr      = polynomial.polynomial(pX, polynomial.eLagr)
            lagrW     = polynomial.polynomial(pX, polynomial.eLagr)
            emax      = 0.0
            plot_tit  = "Basis Degree "+str(pX)
            # ------------------------------------------ #
            # Get maximum error to scale everything
            # ------------------------------------------ #
            for i in range(mesh.theNOfElements):
                t1   = initialMesh.getUElement(i)
                u1   = initialMesh.getXElement(i)
                sol1 = numpy.zeros([len(t1),9])
                a         = t1[0]
                b         = t1[len(t1) - 1]
                for j in range(len(t1)):
                    sol1[j] = f(t1[j])
                    dist    = (sol1[j,0] - u1[j][0]) * (sol1[j,0] - u1[j][0])
                    dist   += (sol1[j,1] - u1[j][1]) * (sol1[j,1] - u1[j][1])
                    emax    = max (numpy.sqrt(dist), emax);
            # ------------------------------------------ #
            d_1 = 0.0
            d_11 = 0
            for i in range(mesh.theNOfElements):

                u0  = initialMesh.getXElement(i)
                u1  = mesh.getXElement(i)
                du1 = mesh.getDNXElement(i)
                disp1 = 0.0

                if (callFix == True) :
                    uf   = meshFix.getXElement(i)
                    tf   = meshFix.getUElement(i)
                    solf = numpy.zeros([len(tf),9])
                    af   = tf[0]
                    bf   = tf[len(tf) - 1]

                t0   = initialMesh.getUElement(i)
                t1   = mesh.getUElement(i)
                sol0 = numpy.zeros([len(t0),9])
                sol1 = numpy.zeros([len(t1),9])
                a1   = t1[0]
                b1   = t1[len(t1) - 1]

                a0   = t0[0]
                b0   = t0[len(t0) - 1]

                (zd, wd)  = quadratures.qType(pX + 1, quadratures.eLGL)
                zzd       = numpy.zeros([pX + 1,1])
                zzd[:, 0] = zd[:]
                T         = numpy.zeros( pX + 1)
                iP        = numpy.zeros([pX + 1,9])
                polyX     = mesh.theMasterElementX.getShapeFunctionsAtPoints(zzd)
                base      = mesh.theNodes[mesh.theElementsX[i,:],:]

                for j in range(pX + 1):
                    T[j]  = 0.5 * (b1 - a1) * zd[j] + 0.5 * (a1 + b1)
                    for d in range(pX + 1):
                        iP[j,0] += polyX[0][j][d] * base[d][0]
                        iP[j,1] += polyX[0][j][d] * base[d][1]
                lagrW.interpolate(zd, iP)

                polyX     = meshFix.theMasterElementX.getShapeFunctionsAtPoints(zzd)
                base      = meshFix.theNodes[meshFix.theElementsX[i,:],:]
                iP        = numpy.zeros([pX + 1,9])
                for j in range(pX + 1):
                    T[j]  = 0.5 * (bf - af) * zd[j] + 0.5 * (af + bf)
                    for d in range(pX + 1):
                        iP[j,0] += polyX[0][j][d] * base[d][0]
                        iP[j,1] += polyX[0][j][d] * base[d][1]
                lagr.interpolate(zd, iP)

                for j in range( len(t0) ):
                    sol0[j] = f(t0[j])

                    dist0   = (sol0[j,0] - u0[j,0]) * (sol0[j,0] - u0[j,0]) + \
                              (sol0[j,1] - u0[j,1]) * (sol0[j,1] - u0[j,1])
                    sol1[j] = f(t1[j])

                    dist1   = (sol1[j,0] - u1[j,0]) * (sol1[j,0] - u1[j,0]) + \
                              (sol1[j,1] - u1[j,1]) * (sol1[j,1] - u1[j,1])

                    tt      = 2.0 * t1[j] - 1.0
                    t       = ortho_project(sol1[j], a1, b1, tt, lagrW, lagrW.eLagr)
                    ulw     = lagrW.evaluate(-1, 1, t)
                    distw   = (sol1[j,0] - ulw[0]) * (sol1[j,0] - ulw[0]) + \
                              (sol1[j,1] - ulw[1]) * (sol1[j,1] - ulw[1])

                    #print(" DISTANCE PROJECTIONS ",dist1 - distw)

                    li[0,ref]  = max(dist0, li[0,ref])
                    li[1,ref]  = max(dist1, li[1,ref])
                    if callFix == True:
                        solf[j] = f(tf[j])
                        distf   = (solf[j,0] - uf[j,0]) * (solf[j,0] - uf[j,0]) + \
                                  (solf[j,1] - uf[j,1]) * (solf[j,1] - uf[j,1])
                        l2[2,ref] += distf * w[j] * 0.5 * (tf[len(tf) - 1] - tf[0])
                        li[2,ref]  = max(distf, li[2,ref])
                        tt      = 2.0 * tf[j] - 1.0
                        t       = ortho_project(solf[j], af, bf, tt, lagrW, lagrW.eLagr)
                        ul      = lagrW.evaluate(-1, 1, t)
                        distl   = (solf[j,0] - ul[0]) * (sol1[j,0] - ul[0]) + \
                                  (solf[j,1] - ul[1]) * (sol1[j,1] - ul[1])

                        #print(" DISTANCE PROJECTIONS NO SIGMA",distf - distl)
                    der1        = numpy.sqrt (du1[j,0] * du1[j,0] + du1[j,1] * du1[j,1])
                    #der1        = numpy.sqrt (ulw[3] * ulw[3] + ulw[4] * ulw[4])
                    disp1      +=  w[j] * dist1 * der1
                    l2 [0,ref] += dist0 * w[j] * 0.5 * (t0[len(t0) - 1] - t0[0])
                    l2 [1,ref] += dist1 * w[j] * 0.5 * (t1[len(t1) - 1] - t1[0])
                d_1   += disp1 * 0.5 * (t1[len(t1) - 1] - t1[0])
                d_11  += disp1# * 0.5 * (t1[len(t1) - 1] - t1[0])
                if ref == nR:
                    if i == 0:
                        fig = plt.figure(2)
                        fig = plt.suptitle(plot_tit)
                        fig = plt.plot(sol1[:,0], sol1[:,1], c = 'c', linestyle='-', linewidth=3, label='exact')
                        fig = plt.plot(u1  [:,0], u1[:,1], c = 'red', linestyle='-', label='Opti')
                        fig = plt.plot(uf  [:,0], uf[:,1], c = 'orange', linestyle='-', label='Opti')
                        fig = plt.legend(loc = 'best')

                        fig = plt.figure(4)
                        fig = plt.suptitle(plot_tit)
                        fig = plt.plot(sol0[:,0], sol0[:,1], c = 'c', linestyle='-', linewidth=3, label='exact')
                        fig = plt.plot(u0  [:,0], u0[:,1], c = 'green', linestyle='-', label='init')
                        fig = plt.legend(loc = 'best')

                        if callFix == True:
                            fig = plt.figure(6)
                            plt.legend(loc = 'best')
                            fig = plt.plot(solf[:,0], solf[:,1], c = 'c', linestyle='-', linewidth=3, label='exact')
                            fig = plt.plot(uf  [:,0], uf[:,1], c = 'orange', linestyle='-', label='Fixed')

                    else:
                        fig = plt.figure(2)
                        fig = plt.suptitle(plot_tit)
                        fig = plt.plot(sol1[:,0], sol1[:,1], c = 'c', linestyle='-', linewidth=3)
                        fig = plt.plot(u1  [:,0], u1[:,1], c = 'red', linestyle='-')

                        fig = plt.figure(4)
                        fig = plt.suptitle(plot_tit)
                        fig = plt.plot(sol0[:,0], sol0[:,1], c = 'c', linestyle='-', linewidth=3)
                        fig = plt.plot(u0  [:,0], u0[:,1], c = 'green', linestyle='-')
                        if callFix == True:
                            fig = plt.figure(6)
                            fig = plt.plot(solf[:,0], solf[:,1], c = 'c', linestyle='-', linewidth=3)
                            fig = plt.plot(uf  [:,0], uf[:,1], c = 'orange', linestyle='-')


                    error0       = (sol0[:, 0] - u0[:,0]) / emax
                    error1       = (sol1[:, 0] - u1[:,0]) / emax + 1.5
                    errorNoScale = (sol1[:, 0] - u1[:,0])
                    if callFix == True:
                        errorf   = (solf[:, 0] - uf[:,0]) / emax + 3.0

                    plot_init = True
                    if (i == 0) :
                        fig = plt.figure(3)
                        fig = plt.suptitle(plot_tit)
                        if (plot_init == True):
                                plt.plot(t0, error0,'-',color='g',linewidth=3.0, label='Init')#-bo')
                        if callFix == True:
                                plt.plot(tf, errorf,'-',color='orange',linewidth=3.0, label='Fix')#-bo')
                        plt.plot(t1, error1,'-',color='r',linewidth=3.0, label='Opt')#-bo')
                        fig = plt.legend(loc = 'best')

                        fig = plt.figure(10)
                        fig = plt.legend(loc = 'best')
                        fig = plt.suptitle(plot_tit)
                        plt.plot( t1, errorNoScale     ,'-',color='g',linewidth=3.0, label = 'Opt')
                    else :
                        fig = plt.figure(3)
                        if (plot_init == True):
                            plt.plot(t0, error0,'-',color='g',linewidth=3.0)
                        if callFix == True:
                            plt.plot(tf, errorf,'-',color='orange',linewidth=3.0)
                        plt.plot(t1, error1,'-',color='r',linewidth=3.0)

                        fig = plt.figure(10)
                        fig = plt.suptitle(plot_tit)
                        plt.plot( t1, errorNoScale     ,'-',color='g',linewidth=3.0)
                    mie = min(errorNoScale)
                    mae = max(errorNoScale)
                    fig = plt.figure(10)
                    tx  = [t1[0], t1[0]]
                    ty  = [mie, mae]
                    fig = plt.plot (tx, ty, color = 'gray',linewidth='0.25', linestyle = '-')
                    tx  = [t1[len(t1) - 1], t1[len(t1) - 1]]
                    fig = plt.plot (tx, ty, color = 'gray',linewidth='0.25', linestyle = '-')

                    ty  = [0,0]
                    tx  = [t1[0], t1[len(t1) - 1] ]
                    fig = plt.plot (tx, ty, color = 'gray',linewidth='0.5', linestyle = '-')

                    tx  = [ t1[ (len(t1) - 1)], t1[ len(t1) - 1] ]
                    fig = plt.plot (tx, ty, color = 'gray',linewidth='0.75', linestyle = '-')

                    if (callFix == True):
                        fig = plt.figure(3)
                        mie = min(errorf)
                        mae = max(errorf)
                        tx  = [tf[0], tf[0]]
                        ty  = [mie, mae]
                        fig = plt.plot (tx, ty, color = 'gray',linewidth='0.25', linestyle = '-')
                        tx  = [tf[len(tf) - 1], tf[len(tf) - 1]]
                        fig = plt.plot (tx, ty, color = 'gray',linewidth='0.25', linestyle = '-')

                        ty  = [3.0,3.0]
                        tx  = [tf[0], tf[len(tf) - 1] ]
                        fig = plt.plot (tx, ty, color = 'gray',linewidth='0.5', linestyle = '-')

                        tx  = [ tf[ (len(tf) - 1)], tf[ len(tf) - 1] ]
                        fig = plt.plot (tx, ty, color = 'gray',linewidth='0.75', linestyle = '-')

                    if (plot_init == True):
                        fig = plt.figure(3)
                        mie = min(error0)
                        mae = max(error0)
                        tx  = [t0[0], t0[0]]
                        ty  = [mie, mae]
                        fig = plt.plot (tx, ty, color = 'gray',linewidth='0.25', linestyle = '-')
                        tx  = [t0[len(t0) - 1], t0[len(t0) - 1]]
                        fig = plt.plot (tx, ty, color = 'gray',linewidth='0.25', linestyle = '-')

                        ty  = [0 ,0]
                        tx  = [t0[0], t0[len(t0) - 1] ]
                        fig = plt.plot (tx, ty, color = 'gray',linewidth='0.5', linestyle = '-')

                        tx  = [ t0[ (len(t0) - 1)], t0[ len(t0) - 1] ]
                        fig = plt.plot (tx, ty, color = 'gray',linewidth='0.75', linestyle = '-')


                    mie = min(error1)
                    mae = max(error1)

                    fig = plt.figure(3)
                    tx  = [t1[0], t1[0]]
                    ty  = [mie, mae]
                    fig = plt.plot (tx, ty, color = 'gray',linewidth='0.25', linestyle = '-')
                    tx  = [t1[len(t1) - 1], t1[len(t1) - 1]]
                    fig = plt.plot (tx, ty, color = 'gray',linewidth='0.25', linestyle = '-')

                    ty  = [1.5 ,1.5]
                    tx  = [t1[0], t1[len(t1) - 1] ]
                    fig = plt.plot (tx, ty, color = 'gray',linewidth='0.5', linestyle = '-')

                    tx  = [ t1[ (len(t1) - 1)], t1[ len(t1) - 1] ]
                    fig = plt.plot (tx, ty, color = 'gray',linewidth='0.75', linestyle = '-')






            domain    = (I[1] - I[0])

            l2[0,ref] = numpy.sqrt(l2[0,ref] / domain)
            l2[1,ref] = numpy.sqrt(l2[1,ref] / domain)
            li[0,ref] = numpy.sqrt(li[0,ref] / domain)
            li[1,ref] = numpy.sqrt(li[1,ref] / domain)
            if (callFix == True):
                l2[2,ref] = numpy.sqrt(l2[2,ref] / domain)
                li[2,ref] = numpy.sqrt(li[2,ref] / domain)
            print( " MY DISPARITY ",d_1, " DOMAIN DISPARITY ", d_1 / domain, "NO SCALE ", d_11)
            disparity1[ref] = d_1

        print("\n\n ------------------- POLYNOMIAL DEGREE ",pX," ---------------\n")
        print ("\t\tDISPARITY\t\t\t  ORDER\t\t\t(ORDER d = sqrt(E))\n")
        print ("N\t  PROJ0\t     OPTI     FIXINT\t   PROJ0    OPTI    FIXINT\tPROJ0   OPTI   FIXINT")
        print("____________________________________________________________________________________________\n")
        for r in range(nR + 1):
            ne1 = pow(2, r) * ne
            if r == 0:
                print (ne1,"\t%1.3e"%disparity0[r]," %1.3e" %disparity1[r]," %1.3e" %disparityF1[r]," |")
            else:
                a = numpy.log10( disparity0[r-1] /  disparity0[r]) / numpy.log10(2.0)
                b = numpy.log10( disparity1[r-1] /  disparity1[r]) / numpy.log10(2.0)
                if (callFix == True):
                    c = numpy.log10(disparityF1[r-1] / disparityF1[r]) / numpy.log10(2.0)
                    f = numpy.log10(numpy.sqrt(disparityF1[r-1]) / numpy.sqrt(disparityF1[r])) / numpy.log10(2.0)
                else:
                    c = 0
                    f = 0
                d = numpy.log10(numpy.sqrt( disparity0[r-1]) / numpy.sqrt( disparity0[r])) / numpy.log10(2.0)
                e = numpy.log10(numpy.sqrt( disparity1[r-1]) / numpy.sqrt( disparity1[r])) / numpy.log10(2.0)
                print (ne1,"\t%1.3e"%disparity0[r]," %1.3e" %disparity1[r]," %1.3e" %disparityF1[r], " |",
                " %1.2f"%a,"   %1.2f" %b,"    %1.2f" %c,"|\t%1.2f"%d,"   %1.2f" %e,"   %1.2f" %f)
        print("____________________________________________________________________________________________\n")
        print("\n----------------------------------  L2 CONVERGENCE  -------------------------------------")
        print ("\t\tError \t\t\t  ORDER\t\t\n")
        print ("N\t  PROJ0\t     OPTI     FIXINT\t   PROJ0    OPTI    FIXINT")
        print("____________________________________________________________________________________________\n")
        for r in range(nR + 1):
            ne1 = pow(2, r) * ne
            if r == 0:
                print (ne1,"\t%1.3e"%l2[0,r]," %1.3e" %l2[1,r]," %1.3e"%l2[2,r]," |")
            else:
                a = numpy.log10(l2[0,r-1] / l2[0,r]) / numpy.log10(2.0)
                b = numpy.log10(l2[1,r-1] / l2[1,r]) / numpy.log10(2.0)
                if (callFix == False): c = 0
                else: c = numpy.log10(l2[2,r-1] / l2[2,r]) / numpy.log10(2.0)
                print (ne1,"\t%1.3e"%l2[0,r]," %1.3e" %l2[1,r]," %1.3e" %l2[2,r]," |",
                " %1.2f"%a,"   %1.2f" %b,"    %1.2f" %c)
        print("_________________________________________________________________________________________\n")
        print("\n--------------------------------  Linfty CONVERGENCE  -----------------------------------")
        print ("\t\tError \t\t\t  ORDER\t\t\n")
        print ("N\t  PROJ0\t     OPTI     FIXINT\t   PROJ0    OPTI    FIXINT")
        print("____________________________________________________________________________________________\n")
        for r in range(nR + 1):
            ne1 = pow(2, r) * ne
            if r == 0:
                print (ne1,"\t%1.3e"%li[0,r]," %1.3e" %li[1,r]," %1.3e" %li[2,r]," |")
            else:
                a = numpy.log10(li[0,r-1] / li[0,r]) / numpy.log10(2.0)
                b = numpy.log10(li[1,r-1] / li[1,r]) / numpy.log10(2.0)
                if (callFix == False): c = 0
                else: c = numpy.log10(li[2,r-1] / li[2,r]) / numpy.log10(2.0)
                print (ne1,"\t%1.3e"%li[0,r]," %1.3e" %li[1,r]," %1.3e" %li[2,r], " |",
                " %1.2f"%a,"   %1.2f" %b,"    %1.2f" %c)
        print("_________________________________________________________________________________________\n")
        plt.show()



if __name__ == '__main__':

    argc = len(sys.argv)
    if argc != 5:
        print (" I NEED DEGREE + INITIAL ELEMENTS + REFINEMENTS + CURVE TYPE")
        quit(1)
    deg    = int(sys.argv[1])  # number of elements
    elmts  = int(sys.argv[2])  # number of elements
    refine = int(sys.argv[3])  # number of elements
    curve  = int(sys.argv[4])  # number of elements

    I = [-1.5, 1.5]

    if (curve == 1) :
        print(" APPROXIMATING A QUADRATIC FUNCTION")
        def  f(t): return (  t,       (3.0 * t + 1.0) * (3.0 * t + 1.0), 0.0,
                           1.0, 2.0 * (3.0 * t + 1.0) *  3.0, 0.0,
                           0.0, 18.0, 0.0)

    elif (curve == 2) :
       print(" APPROXIMATING A CUBIC FUNCTION")
       def  f(t): return ( t, t * t * t + 3.0 * t , 0.0,
                           1.0, 3.0 * t * t + 3.0, 0.0,
                           0.0, 6.0 * t, 0.0)
    elif (curve == 3) :
       print(" APPROXIMATING A QUARTIC FUNCTION")
       def  f(t): return ( t,         t * t * t * t + 3.0 * t * t * t + 2.0 * t ,0.0,
                           1.0, 4.0 * t * t * t     + 9.0 * t * t     + 2.0     ,0.0,
                           0.0,12.0 * t * t         + 18.0 * t                  ,0.0)

    elif (curve == 4):
        I = [0, 25]
        print(" APPROXIMATING A SPIRAL FUNCTION")

        def  f(t):
            a   = numpy.exp(t / 10.0)
            b   = [ 0.1 * a * numpy.cos(t) ,  0.1 * a * numpy.sin (t), 0.0]
            df  = [b[0] - a * numpy.sin(t) , b[1] + a * numpy.cos (t), 0.0]
            b   = [(0.01 - 1.0) * a * numpy.cos(t), (0.01 - 1.0) * a * numpy.sin(t), 0.0]
            ddf = [b[0] - 0.1 * a * numpy.sin(t), b[1] + 0.1 * a * numpy.cos(t), 0.0]
            return ( a * numpy.cos(t) , a * numpy.sin (t), 0.0,
                    df[0], df[1], df[2], ddf[0], ddf[1],ddf[2])

    elif (curve == 5):
        I = [0, 0.5*numpy.pi]
        print(" APPROXIMATING A CIRCULAR ARC")

        def  f(t):return ( numpy.cos(t), numpy.sin(t), 0.0,
                          -numpy.sin(t), numpy.cos(t), 0.0,
                          -numpy.cos(t), numpy.sin(t), 0.0)
    elif (curve == 6) :
       print(" APPROXIMATING A POLYNOMIAL OF DEGREE 6")
       def  f(t): return ( t,         pow(t,6) + 3.0  * t * t * t + 3.0 * t * t + 1, 0.0,
                           1.0, 6.0 * pow(t,5) + 9.0  * t * t     + 6.0 * t        , 0.0,
                           0.0,30.0 * pow(t,4) + 18.0 * t         + 6.0            , 0.0)
    elif (curve == 7) :
    #   I = [0, 0.5*numpy.pi]
       print(" APPROXIMATING A SINE WAVE")
       def  f(t): return (   t,  numpy.sin(t), 0.0,
                           1.0, -numpy.cos(t), 0.0,
                           0.0, -numpy.sin(t), 0.0)
    TestDistanceFunctionOptimization.testDistanceFunction(deg, elmts, refine, curve, f, I)
