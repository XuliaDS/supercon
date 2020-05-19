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



from Discretization.Meshers import CurveMesher,SurfaceMesher

from Writers.NumpyMeshWriter import NumpyMeshWriter

from Globals.configPython import *

import polynomial
import quadratures



class referenceMesh(Curve.Curve):

    def __init__(self, a, b, mesh):
        Curve.Curve.__init__(self,1)

        self.theT0    = a
        self.theT1    = b
        self.theMesh  = mesh

    def value(self,t):

        a = self.theT0
        b = self.theT1
        z = 2.0 / (b - a) * (t - (a + b) * 0.5)
        polyX = self.theMesh.theMasterElementX.getShapeFunctionsAtPoints(z)
        base  = self.theMesh.theNodes[self.theMesh.theElementsX[0,:],:]
        value = numpy.zeros([len(t),1])
        for j in range(len(t)):
            for d in range(self.theMesh.theDegreeX + 1):
                value[j] += polyX[0][j][d] *  base[d][0]

        return value




class TestDistanceFunctionOptimization(unittest.TestCase):

    @staticmethod


    def getGeometry1D(c, a, b):
        if   c ==  0: return Curve1DPol.Curve1DCos(a, b)
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

        relocateX = False
        fixU      = False
        callFix   = True
        method    = 'Newton'
        tolDistanceCalculation = 1.e-7

        tol = 1.e-7

        if curve == 3: inverseMap = True
        else: inverseMap = False
        dis = numpy.zeros([2, nR + 1])
        li  = numpy.zeros([2, nR + 1])

        nOfSubdivisions       = 25
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        disp_t   = numpy.zeros(nR + 1)
        dis_FixX = numpy.zeros(nR + 1)
        parametrization = TestDistanceFunctionOptimization.getGeometry1D(curve, I[0], I[1])
        h0              = (I[1] - I[0])

        for ref in range(nR + 1):
            n = pow (2, ref) * ne
            h = (I[1] - I[0]) / n

            optimizer = Optimization.DistanceFunction.DistanceFunctionOptimization.DistanceFunctionOptimization(
                parametrization,
                h,pX,pU,
                objectiveFunctionName,
                tol,
                initialP  = pX,
                method    = method,
                relocateX = relocateX,
                fixU      = True
                )

            mesh, fixedMesh  = optimizer.run()

            w = fixedMesh.theMasterElementX.theGaussWeights
            z = fixedMesh.theMasterElementX.theGaussPoints

            if (ref == 0):
                baseMesh = CurveMesher.CurveMesher(parametrization, h0, pX, pU, len(w) - 1)
                baseMesh.run()
                fixPar   = referenceMesh(I[0], I[1], baseMesh.theMesh)

            refineMesh = CurveMesher.CurveMesher(fixPar, h, pX, pU, len(w) - 1)
            refineMesh.run()

            for i in range(fixedMesh.theNOfElements):
                fixedMesh.theElementsX[i,:]                  = refineMesh.theMesh.theElementsX[i,:]
                fixedMesh.theNodes[mesh.theElementsX[i,:],:] = refineMesh.theMesh.theNodes[refineMesh.theMesh.theElementsX[i,:],:]

            disf,proje,norm = TestDistanceFunctionOptimization.getMeshDistances(
                                    fixedMesh,parametrization,frechetFunctionName,
                                    tolDistanceCalculation, nOfSubdivisions)

            meshO, meshI   = optimizer.run()


            if (ref == nR):
                fig = plt.figure(10)
                for i in range(fixedMesh.theNOfElements):
                    uO  = fixedMesh.getXElement(i)
                    tO  = fixedMesh.getUElement(i)

                    uf  = refineMesh.theMesh.getXElement(i)
                    tf  = refineMesh.theMesh.getUElement(i)

                    zzf = 0.5 * (tf[-1] - tf[0]) * z + 0.5 * (tf[-1] + tf[0])
                    zz1 = 0.5 * (tO[-1] - tO[0]) * z + 0.5 * (tO[-1] + tO[0])

                    plt.plot(tf, uf, c = 'red', linewidth = 2)
                    plt.plot(tO, uO, c = 'orange', linestyle = ':')

                    plt.plot(zzf, uf, c = 'b', linewidth = 2)
                    plt.plot(zz1, uO, c = 'c', linestyle = ':')

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

            meshO, meshI = optimizer.run()
            disp,proj,norm = TestDistanceFunctionOptimization.getMeshDistances(
                                meshI,parametrization,frechetFunctionName,
                                tolDistanceCalculation, nOfSubdivisions)

            dis[0,ref]    = disp * disp * 0.5

            disp,proj,norm = TestDistanceFunctionOptimization.getMeshDistances(
                                meshO,parametrization,frechetFunctionName,
                                tolDistanceCalculation, nOfSubdivisions)

            dis[1,ref]    = disp * disp * 0.5
            dis_FixX[ref] = disf * disf * 0.5

            #print(" DISPARITY OPTIMIZED",dis[1,ref], "FIXED ",dis_FixX[ref])
            plot_tit  = "Basis Degree " + str(pX)

            '''
            uf0 = fixmeshI.getXElement(0)
            tf0 = fixmeshI.getUElement(0)
            fig = plt.figure(1)
            plt.plot(tf0, uf0, c = 'c',   linestyle='-', linewidth=7, label='Master')

            for i in range(mesh.theNOfElements):
                uf    = fixMesh1.getXElement(i)
                tf    = fixMesh1.getUElement(i)
                solf  = parametrization.value(tf)
                plt.plot(tf, uf,  label='Refine')
            plt.show()
            '''

            for i in range(mesh.theNOfElements):

                uf   = fixedMesh.getXElement(i)
                tf   = fixedMesh.getUElement(i)
                solf = parametrization.value(tf)

                uO   = meshO.getXElement(i)
                tO   = meshO.getUElement(i)
                solO = parametrization.value(tO)

                uI   = meshI.getXElement(i)
                tI   = meshI.getUElement(i)
                solI = parametrization.value(tI)



                if (inverseMap == True):
                    inv1  = parametrization.inv_value(uO)
                    eleID = meshO.theElementsX[i,:]
                    dMesh = meshO.theMasterElementX.getDXElement( meshO.theNodes[eleID,:] )
                    dist  = 0.0
                    for j in range(len(tO)):
                        dist += (tO[j] - inv1[j]) * (tO[j] - inv1[j]) * abs(dMesh[j]) * w[j]
                    disp_t[ref] += dist

                zz1 = 0.5 * (tO[-1] - tO[0]) * z + 0.5 * (tO[-1] + tO[0])
                zzf = 0.5 * (tf[-1] - tf[0]) * z + 0.5 * (tf[-1] + tf[0])

                if ref == nR:
                    if i == 0:
                        fig = plt.figure(2)
                        plt.suptitle(plot_tit)
                        plt.plot(tO, solO, c = 'c',   linestyle='-', linewidth=4, label='Exact tOpt')
                        plt.plot(zz1,   uO, c = 'red', linestyle='-', linewidth=3, label='Opti')

                        plt.plot(tf, solf, c = 'b'  , linestyle='-.', linewidth=3, label='Exact tfix')
                        plt.plot(zzf,   uf, c = 'orange', linestyle='-.', linewidth=2, label='x_fix')


                        plt.legend(loc = 'best')

                        fig = plt.figure(3)
                        plt.plot(zz1, solO, c = 'c'  , linestyle='-', linewidth=4, label='exact')
                        plt.plot(zz1, uO  , c = 'red', linestyle='-', linewidth=3, label='Opti')

                        plt.plot(zzf, solf, c = 'b'  , linestyle='-.', linewidth=3, label='Exact tfix')
                        plt.plot(zzf, uf  , c = 'orange', linestyle='-.', linewidth=2, label='x_fix')

                        plt.legend(loc = 'best')

                    else:
                        fig = plt.figure(2)
                        plt.plot(tO, solO, c = 'c',   linestyle='-', linewidth=4)
                        plt.plot(tO,   uO, c = 'red', linestyle='-', linewidth=3)

                        plt.plot(tf, solf, c = 'b'  , linestyle='-.', linewidth=3)
                        plt.plot(tf,   uf, c = 'orange', linestyle='-.', linewidth=2)

                        fig = plt.figure(3)
                        plt.suptitle(plot_tit)
                        plt.plot(zz1, solO, c = 'c'  , linestyle='-', linewidth=4)
                        plt.plot(zz1, uO  , c = 'red', linestyle='-', linewidth=3)

                        plt.plot(zzf, solf, c = 'b'  , linestyle='-.', linewidth=3)
                        plt.plot(zzf, uf  , c = 'orange', linestyle='-.', linewidth=2)
                    print(" END INTERVALS X", solI[0], solI[-1]," MESH", uI[0], uI[-1], " ERROR ", abs(solI[0] - uI[0]), abs(solI[-1] - uI[-1]))
                    fig = plt.figure(30)
                    plt.plot(tI, solI - uI, c = 'b'  , linestyle='-.', linewidth=3)
                    plt.scatter(tI[0], 0, c = 'r')
                    plt.scatter(tI[-1], 0, c = 'r')

        print("------------------------------------------------------------------------")
        print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pU," ----------------")
        print("------------------------------------------------------------------------\n")

        if (inverseMap == True):
            print("\n \t|| t_q - t^* || ==> expect p + q =", pX + pU,"\n\n")
            print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
            print("------------------------------------------------------------------------\n")
            for r in range(nR + 1):
                ne1 = pow(2, r) * ne
                if r == 0:
                    print (ne1,"\t%1.3e"%disp_t[r],"         |","%1.3e"%numpy.sqrt(disp_t[r]))
                else:
                    a = numpy.log10(           disp_t[r-1] /             disp_t[r]) / numpy.log10(2.0)
                    b = numpy.log10(numpy.sqrt(disp_t[r-1])/ numpy.sqrt(disp_t[r])) / numpy.log10(2.0)
                    print (ne1,"\t%1.3e"%disp_t[r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(disp_t[r]),"  %1.2f" %b)
            print("____________________________________________________________________\n")

        print("\n \t|| xfix_p - alpha t_q^* || ==> expect q + 1 = ", pU + 1,"\n\n")
        print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
        print("------------------------------------------------------------------------\n")
        for r in range(nR + 1):
            ne1 = pow(2, r) * ne
            if r == 0:
                print (ne1,"\t%1.3e"%dis_FixX[r],"         |","%1.3e"%numpy.sqrt(dis_FixX[r]))
            else:
                a = numpy.log10(           dis_FixX[r-1] /            dis_FixX[r]) / numpy.log10(2.0)
                b = numpy.log10(numpy.sqrt(dis_FixX[r-1])/ numpy.sqrt(dis_FixX[r])) / numpy.log10(2.0)
                print (ne1,"\t%1.3e"%dis_FixX[r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(dis_FixX[r]),"  %1.2f" %b)
        print("____________________________________________________________________\n")


        print("\n \t|| x_p^* - alpha t_q || ==>  expect p + q =", pX + pU,"\n\n")
        print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
        print("------------------------------------------------------------------------\n")
        for r in range(nR + 1):
            ne1 = pow(2, r) * ne
            if r == 0:
                print (ne1,"\t%1.3e"%dis[1,r],"         |","%1.3e"%numpy.sqrt(dis[1,r]))
            else:
                a = numpy.log10(           dis[1,r-1] /            dis[1,r]) / numpy.log10(2.0)
                b = numpy.log10(numpy.sqrt(dis[1,r-1])/ numpy.sqrt(dis[1,r])) / numpy.log10(2.0)
                print (ne1,"\t%1.3e"%dis[1,r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(dis[1,r]),"  %1.2f" %b)
        print("____________________________________________________________________\n")
        print("\n \t|| I_p^* - alpha t_q || ==>  expect p + q =", pX + pU,"\n\n")
        print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
        print("------------------------------------------------------------------------\n")
        for r in range(nR + 1):
            ne1 = pow(2, r) * ne
            if r == 0:
                print (ne1,"\t%1.3e"%dis[0,r],"         |","%1.3e"%numpy.sqrt(dis[0,r]))
            else:
                a = numpy.log10(           dis[0,r-1] /            dis[0,r]) / numpy.log10(2.0)
                b = numpy.log10(numpy.sqrt(dis[0,r-1])/ numpy.sqrt(dis[0,r])) / numpy.log10(2.0)
                print (ne1,"\t%1.3e"%dis[0,r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(dis[0,r]),"  %1.2f" %b)
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
    if ( curve == 0): I = [0, numpy.pi]
    else:             I = [0.25, 1]

    TestDistanceFunctionOptimization.testDistanceFunction(degX, degT, elmts, refine, curve, I, showPlots)
