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


def euc_dist (a, b, dim):
    c = b - a
    return norm(c,dim)

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
        if   c == 4: return Curve2DPol.Curve2DExp (a, b)
        elif c == 1: return Curve2DPol.Curve2DSine(a, b)
        elif c == 11:return Curve2DPol.Curve2DSineSine(a, b)
        elif c == 2: return Curve2DPol.Curve2DPol2(a, b)
        elif c == 3: return Curve2DPol.Curve2DPol3(a, b)
        elif c == 0 or c == 10: return Circle.Circle (1.0, a, b) #Curve2DPol.Curve2DCircle(a,b)
        elif c == 5: return CirclePolynomial.CirclePolynomial(1, 2)

        elif c == 6: return Curve2DPol.Curve2DPol6(a, b)

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
        tolDistanceCalculation = 1.e-10

        tol = 1.e-10

        if curve == 0: inverseMap = True
        else:          inverseMap = False

        nOfSubdivisions       = 50
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        parametrization       =  TestDistanceFunctionOptimization.getGeometry2D(curve, I[0], I[1])
        frechet  = numpy.zeros([2,nR + 1])
        d_xs     = numpy.zeros([2,nR + 1])
        d_xse    = numpy.zeros([2,nR + 1])
        d_sse    = numpy.zeros([2,nR + 1])
        d_t      = numpy.zeros([2,nR + 1])
        dot_con1 = numpy.zeros([2,nR + 1])
        dot_con2 = numpy.zeros([2,nR + 1])

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

            w = meshO.theMasterElementX.theGaussWeights
            z = meshO.theMasterElementX.theGaussPoints

            for type in range(2):
                if (type == 0): mesh = meshO
                else          : mesh = meshI

                disp,proj,norm = TestDistanceFunctionOptimization.getMeshDistances(
                                        mesh,parametrization,frechetFunctionName,
                                        tolDistanceCalculation, nOfSubdivisions)
                frechet[type][ref] = disp * disp * 0.5
                emax0 = 0.0
                emax1 = 0.0
                emaxd = 0.0
                for i in range(mesh.theNOfElements):
                    x = mesh.getXElement(i)
                    t = mesh.getUElement(i)
                    s = parametrization.value(t)
                    for j in range (len(t)):
                        d0 = abs(s[j,0] - x[j,0])
                        d1 = abs(s[j,1] - x[j,1])
                        emax0 = max (emax0, d0)
                        emax1 = max (emax1, d1)
                        emaxd = max (emaxd, numpy.sqrt(d0 * d0 + d1 * d1))


                for i in range(mesh.theNOfElements):

                    x  = mesh.getXElement(i)
                    t  = mesh.getUElement(i)
                    s  = parametrization.value(t)
                    tE =  orthogonal_projection(x, t, parametrization)
                    se = parametrization.value(tE)
                    dt = numpy.einsum('li,klj->kij',
                            mesh.theParametricNodes[mesh.theElementsU[i, :], :],
                            mesh.theMasterElementU.theShapeFunctionsDerivatives)
                    sumXS   = 0.0
                    sumXSE  = 0.0
                    sumSSE  = 0.0
                    sumT    = 0.0
                    dx      = mesh.getDNXElement(i)
                    dalpha  = parametrization.tangent(t)
                    dalphaE = parametrization.tangent(tE)
                    err_px  = numpy.zeros(len(z))
                    err_pa  = numpy.zeros(len(z))
                    err_xE  = numpy.zeros(len(z))
                    err_aE  = numpy.zeros(len(z))
                    pro_xvsa = numpy.zeros(len(z))
                    pro_xvse = numpy.zeros(len(z))

                    a0      = numpy.zeros([len(z),2])
                    a1      = numpy.zeros([len(z),2])
                    a2      = numpy.zeros([len(z),2])
                    a3      = numpy.zeros([len(z),2])
                    a4      = numpy.zeros([len(z),2])
                    n0      = numpy.zeros(len(z))
                    n1      = numpy.zeros(len(z))
                    n2      = numpy.zeros(len(z))

                    dn0     = numpy.zeros(len(z))
                    dn1     = numpy.zeros(len(z))
                    dn2     = numpy.zeros(len(z))
                    SP1     = 0.0
                    SP2     = 0.0
                    for j in range(len(t)):
                        a0[j]     = s[j]  - x[j]
                        a1[j]     = se[j] - x[j]
                        a2[j]     = s[j]  - se[j]
                        err_px[j] = numpy.dot( a0[j], dx[j])
                        err_pa[j] = numpy.dot(-a0[j], dalpha[j])
                        err_xE[j] = numpy.dot( a1[j], dx[j])
                        err_aE[j] = numpy.dot(-a1[j], dalphaE[j])
                        pro_xvsa[j] = numpy.dot( s[j] -x[j],dx[j] -  dalpha[j])
                        pro_xvse[j] = numpy.dot(se[j] -x[j],dx[j] - dalphaE[j])
                        et      = (t[j] - tE[j]) * (t[j] - tE[j])
                        dsse    = numpy.dot (s[j] - se[j], s[j] - se[j])
                        dxs     = numpy.dot (x[j] - se[j], x[j] - se[j])
                        dxse    = numpy.dot (x[j] -  s[j], x[j] - s [j])
                        arc     = dx[j,0,0] * dx[j,0,0] + dx[j,1,0] * dx[j,1,0]
                        der     = numpy.sqrt(arc)
                        sumXS  += der *  dxs * w[j]
                        sumXSE += der * dxse * w[j]
                        #der     = abs(dt[j,0,0])
                        sumSSE += der * dsse * w[j]
                        sumT   += der *   et * w[j]

                        n0[j]   = mynorm(a0[j], 2)
                        n1[j]   = mynorm(a1[j], 2)
                        n2[j]   = mynorm(a2[j], 2)
                        dn0[j]   = mynorm(dalpha[j], 2)
                        dn1[j]   = mynorm(dalphaE[j], 2)
                        dn2[j]   = mynorm(dx[j], 2)
                        SP1    += pro_xvsa[j] * der * w[j]
                        SP2    += pro_xvse[j] * der * w[j]
                    d_xs [type, ref] += sumXS
                    d_xse[type, ref] += sumXSE
                    d_sse[type, ref] += sumSSE
                    d_t  [type, ref] += sumT
                    dot_con1[type, ref] += SP1
                    dot_con2[type, ref] += SP2
                    zx          = z + 2.0 * i
                    strdeg = 'pX '+str(pX)+' pT '+str(pU)
                    if type == 1: tit = ("INTERPOLATION "+ strdeg)
                    else:         tit = ("OPTIMIZED " + strdeg)
                    if ref == nR:
                        if i == meshO.theNOfElements-1:
                            leg_alpha = 'alpha'
                            leg_alphe = 'alpha-E'
                            leg_x     = 'x'
                            leg_t     = 't(xi)'
                            leg_xi    = 'xi(t)'
                            leg_e1    = '1st co'
                            leg_e2    = '2nd co'
                            leg_e3    = '1st co-tE'
                            leg_e4    = '2nd co-tE'
                            leg_d     = 'dist_e'
                        else:
                            leg_alpha = None
                            leg_alphe = None
                            leg_x     = None
                            leg_t     = None
                            leg_xi    = None
                            leg_e1    = None
                            leg_e2    = None
                            leg_e3    = None
                            leg_e4    = None
                            leg_d     = None
                        plotcount = 1
                        '''fig = plt.figure(10 * type + plotcount)
                        plotcount +=1
                        plt.title(tit)
                        plt.plot(zx, dt[:,0,0], c = 'c'  , linestyle='-', label=leg_alpha)
                        if i == meshO.theNOfElements-1: plt.legend(loc = 'best')

                        fig = plt.figure(10 * type + plotcount)
                        plotcount +=1
                        plt.suptitle(tit)
                        plt.plot(zx, pro_xvsa, c = 'b'  , linestyle='-', label=leg_alpha) #  s - x
                        plt.scatter(zx[0], pro_xvsa[0], c = 'g')
                        plt.scatter(zx[-1],pro_xvsa[-1], c = 'g')
                        plt.plot(zx, pro_xvse, c = 'r'  , linestyle=':', label=leg_alphe)     # se - x
                        plt.scatter(zx[0], pro_xvse[0], c = 'g')
                        plt.scatter(zx[-1], pro_xvse[-1], c = 'g')
                        if i == meshO.theNOfElements-1: plt.legend(loc = 'best')
                        fig = plt.figure(10 * type + plotcount)
                        plotcount +=1
                        plt.suptitle(tit)
                        plt.subplot(2,2,1)
                        plt.title(" exact - x, x' ")
                        plt.plot(zx, err_xE, c = 'r'  , linestyle='-')
                        plt.scatter(zx[0], err_xE[0], c = 'b')
                        plt.scatter(zx[-1], err_xE[-1], c = 'b')
                        plt.subplot(2,2,2)
                        plt.title(" alpha -x, x' ")
                        plt.plot(zx, err_px, c = 'r'  , linestyle='-')
                        plt.scatter(zx[0], err_px[0], c = 'b')
                        plt.scatter(zx[-1], err_px[-1], c = 'b')
                        plt.subplot(2,2,3)

                        plt.title(" x - exact, exact' ")
                        plt.plot(zx, err_aE, c = 'r'  , linestyle=':')
                        plt.scatter(zx[0], err_aE[0], c = 'b')
                        plt.scatter(zx[-1], err_aE[-1], c = 'b')

                        plt.subplot(2,2,4)
                        plt.title(" x - alpha, alpha' ")

                        plt.plot(zx, err_pa, c = 'r'  , linestyle=':')
                        plt.scatter(zx[0], err_pa[0], c = 'b')
                        plt.scatter(zx[-1], err_pa[-1], c = 'b')
                        fig = plt.figure(10 * type + plotcount)
                        plotcount +=1
                        plt.suptitle(tit)
                        plt.plot(zx, n0, c = 'b'  , linestyle='-', label=leg_alpha) #  s - x
                        plt.plot(zx, n1, c = 'r'  , linestyle=':', label=leg_x)     # se - x
                        plt.plot(zx, n2, c = 'g'  , linestyle=':', label=leg_alphe) # se - s
                        if i == meshO.theNOfElements-1: plt.legend(loc = 'best')
                        fig = plt.figure(10 * type + plotcount)
                        plotcount +=1
                        plt.suptitle(tit)
                        plt.plot(zx, dn0, c = 'b'  , linestyle='-', label=leg_alpha) #  s - x
                        plt.plot(zx, dn1, c = 'r'  , linestyle=':', label=leg_x)     # se - x
                        plt.plot(zx, dn2, c = 'g'  , linestyle=':', label=leg_alphe) # se - s
                        if i == meshO.theNOfElements-1: plt.legend(loc = 'best')'''
                        err0 = ( s[:,0] - x[:,0]) / emax0
                        err1 = ( s[:,1] - x[:,1]) / emax1
                        err2 = (se[:,0] - x[:,0]) / emax0
                        err3 = (se[:,1] - x[:,1]) / emax1

                        errd = numpy.sqrt((s[:,0] - x[:,0]) * (s[:,0] - x[:,0]) + (s[:,1] - x[:,1]) * (s[:,1] - x[:,1])) / emaxd



                        fig = plt.figure(10 * type +plotcount)
                        plotcount += 1
                        plt.suptitle(tit)

                        plt.plot(s[:,0], s[:,1], c = 'c'  , linestyle='-', label=leg_alpha)
                        plt.plot(x[:,0], x[:,1], c = 'red', linestyle='-', label=leg_x)
                        if i == meshO.theNOfElements-1: plt.legend(loc = 'best')
                        plt.axis('equal')
                        for k in range(2):
                            if k == 0:
                                y    = zx
                                xlab = 'xi'
                            else:
                                y    = t
                                xlab = 't'
                            fig = plt.figure(10 * type +plotcount)
                            plotcount += 1
                            plt.suptitle(tit)

                            plt.subplot(1,2,1)
                            plt.title(' First component')
                            plt.plot(y, s[:,0],c = 'c', linestyle='-',  label=leg_alpha)
                            plt.plot(y, x[:,0],c = 'b', linestyle='-.',  label=leg_x)
                            plt.xlabel(xlab)
                            if i == meshO.theNOfElements-1: plt.legend(loc = 'best')
                            plt.subplot(1,2,2)
                            plt.title(' Second component')
                            plt.plot(y, s[:,1],c = 'c', linestyle='-', label=leg_alpha)
                            plt.plot(y, x[:,1],c = 'b', linestyle='-.', label=leg_x)
                            plt.xlabel(xlab)
                            if i == meshO.theNOfElements-1: plt.legend(loc = 'best')
                            plt.axis('equal')

                        fig = plt.figure(10 * type +plotcount)
                        plotcount += 1

                        plt.title(tit)
                        plt.suptitle(' Error ' )

                        plt.plot(zx, err0, c = 'b', linestyle='-', label=leg_e1)
                        plt.plot(zx, err1, c = 'r', linestyle='-', label=leg_e2)
                        plt.plot(zx, err2, c = 'c', linestyle='-', label=leg_e3)
                        plt.plot(zx, err3, c = 'g', linestyle='-', label=leg_e4)
                        plt.plot(zx, errd, c = 'orange', linestyle='--', label= leg_d)
                        if i == meshO.theNOfElements-1: plt.legend(loc = 'best')
                        T = [zx[0], zx[-1]]
                        Y = [0, 0]
                        plt.plot(T, Y, c='gray', linewidth=1)
                        plt.scatter(T, Y, c = 'orange', s  = 3)
                        fig = plt.figure(10 * type +plotcount)
                        plotcount += 1
                        plt.title(tit)
                        plt.plot(x, t)


        for  type in range(2):
            if type == 0: print('OPTIMIZED MESH')
            else:         print('INTERPOLATIVE INITIAL MESH')
            print("------------------------------------------------------------------------")
            print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pU," ----------------")
            print("------------------------------------------------------------------------\n")

            print("\n \t DISPARITY (FRECHET)==>  expect 2p =", 2 * pX,"\n\n")
            print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%frechet[type][r],"         |","%1.3e"%numpy.sqrt(frechet[type][r]))
                else:
                    a = numpy.log10(           frechet[type][r-1] /            frechet[type][r]) / numpy.log10(2.0)
                    b = numpy.log10(numpy.sqrt(frechet[type][r-1])/ numpy.sqrt(frechet[type][r])) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%frechet[type][r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(frechet[type][r]),"  %1.2f" %b)
            print("____________________________________________________________________\n")
            print("\n \t DISPARITY (EU)==>  expect 2p =", 2 * pX,"\n\n")
            print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%d_xs[type][r],"         |","%1.3e"%numpy.sqrt(d_xs[type][r]))
                else:
                    a = numpy.log10(           d_xs[type][r-1] /            d_xs[type][r]) / numpy.log10(2.0)
                    b = numpy.log10(numpy.sqrt(d_xs[type][r-1])/ numpy.sqrt(d_xs[type][r])) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%d_xs[type][r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(d_xs[type][r]),"  %1.2f" %b)
            print("____________________________________________________________________\n")

            print("\n \t DISPARITY T EXACT ==>  expect 2p =", 2 * pX,"\n\n")
            print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%d_xse[type][r],"         |","%1.3e"%numpy.sqrt(d_xse[type][r]))
                else:
                    a = numpy.log10(           d_xse[type][r-1] /            d_xse[type][r]) / numpy.log10(2.0)
                    b = numpy.log10(numpy.sqrt(d_xse[type][r-1])/ numpy.sqrt(d_xse[type][r])) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%d_xse[type][r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(d_xse[type][r]),"  %1.2f" %b)
            print("____________________________________________________________________\n")

            print("\n \t DISPARITY PROJECTION ==>  expect 4p =", 4 * pX,"\n\n")
            print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%d_sse[type][r],"         |","%1.3e"%numpy.sqrt(d_sse[type][r]))
                else:
                    a = numpy.log10(           d_sse[type][r-1] /            d_sse[type][r]) / numpy.log10(2.0)
                    b = numpy.log10(numpy.sqrt(d_sse[type][r-1])/ numpy.sqrt(d_sse[type][r])) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%d_sse[type][r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(d_sse[type][r]),"  %1.2f" %b)
            print("____________________________________________________________________\n")
            print("\n \t DISPARITY T PARAMETRIZATION ==>  expect 4p =", 4 * pX,"\n\n")
            print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%d_t[type][r],"         |","%1.3e"%numpy.sqrt(d_t[type][r]))
                else:
                    a = numpy.log10(           d_t[type][r-1] /            d_t[type][r]) / numpy.log10(2.0)
                    b = numpy.log10(numpy.sqrt(d_t[type][r-1])/ numpy.sqrt(d_t[type][r])) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%d_t[type][r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(d_t[type][r]),"  %1.2f" %b)
            print("____________________________________________________________________\n")
            print("\n \t CONVERTENCE CON1 CON2  ==>  expect 2p + 1=", 2 * pX + 1,"\n\n")
            print ("N\t ALPHA\t    ORDER    EXACT    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%dot_con1[type][r],"         |","%1.3e"%dot_con2[type][r])
                else:
                    a = numpy.log10(dot_con1[type][r-1] / dot_con1[type][r]) / numpy.log10(2.0)
                    b = numpy.log10(dot_con2[type][r-1] / dot_con2[type][r]) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%dot_con1[type][r],"  %1.2f"%a, "  | %1.3e"%dot_con2[type,r],"  %1.2f" %b)
            print("____________________________________________________________________\n")
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
    else: I = [0.25, 1]

    TestDistanceFunctionOptimization.testDistanceFunction(degX, degT, elmts, refine, curve, I, showPlots)
