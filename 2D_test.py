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




class TestDistanceFunctionOptimization(unittest.TestCase):

    @staticmethod


    def getGeometry2D(c, a, b):
        if   c ==  0: return Curve2DPol.Curve2DExp (a, b)
        elif c ==  1: return Curve2DPol.Curve2DSine(a, b)
        elif c ==  11:return Curve2DPol.Curve2DSineSine(a, b)
        elif c ==  2: return Curve2DPol.Curve2DPol2(a, b)
        elif c ==  3: return Curve2DPol.Curve2DPol3(a, b)

        elif c ==  6: return Curve1DPol.Curve2DPol6(a, b)

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
        tolDistanceCalculation = 1.e-8

        tol = 1.e-8

        if curve == 0: inverseMap = True
        else:          inverseMap = False

        nOfSubdivisions       = 25
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        parametrization       =  TestDistanceFunctionOptimization.getGeometry2D(curve, I[0], I[1])
        disp_TI  = numpy.zeros(nR + 1)
        disp_TO  = numpy.zeros(nR + 1)
        disp_I   = numpy.zeros(nR + 1)
        disp_O   = numpy.zeros(nR + 1)

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
                fixU      = fixU
                )


            meshO, meshI   = optimizer.run()
            
            disp,proj,norm = TestDistanceFunctionOptimization.getMeshDistances(
                                            meshI,parametrization,frechetFunctionName,
                                            tolDistanceCalculation, nOfSubdivisions)
            disp_I[ref]    = disp * disp * 0.5

            disp,proj,norm = TestDistanceFunctionOptimization.getMeshDistances(
                                            meshO,parametrization,frechetFunctionName,
                                            tolDistanceCalculation, nOfSubdivisions)
            disp_O[ref]    = disp * disp * 0.5


            w = meshO.theMasterElementX.theGaussWeights
            z = meshO.theMasterElementX.theGaussPoints

            for i in range(meshO.theNOfElements):

                uO   = meshO.getXElement(i)
                tO   = meshO.getUElement(i)
                solO = parametrization.value(tO)

                uI   = meshI.getXElement(i)
                tI   = meshI.getUElement(i)
                solI = parametrization.value(tI)


                zO   = 0.5 * (tO[-1] - tO[0]) * z + 0.5 * (tO[-1] + tO[0])
                zI   = 0.5 * (tI[-1] - tI[0]) * z + 0.5 * (tI[-1] + tI[0])

                plot_tit  = "Basis Degree " + str(pX)
                if ref == nR:
                    fig = plt.figure(20)
                    if i == meshO.theNOfElements -1:

                        plt.suptitle(plot_tit)
                        plt.plot(solO[:,0], solO[:,1], c = 'c'  , linestyle='-', linewidth=4, label='Exact tOpt')
                        plt.plot(uO[:,0], uO[:,1], c = 'red', linestyle='-', linewidth=3, label='Opti')

                        plt.plot(solI[:,0], solI[:,1], c = 'b'     , linestyle='-.', linewidth=3, label='Exact Interp')
                        plt.plot(solI[:,0], solI[:,1], c = 'orange', linestyle='-.', linewidth=2, label='Interp')
                        plt.legend(loc = 'best')

                    else:

                        plt.plot(solO[:,0], solO[:,1], c = 'c'  , linestyle='-', linewidth=4)
                        plt.plot(  uO[:,0], uO[:,1], c = 'red', linestyle='-', linewidth=3)

                        plt.plot(solI[:,0], solI[:,1], c = 'b'     , linestyle='-.', linewidth=3)
                        plt.plot(  uI[:,0],   uI[:,1], c = 'orange', linestyle='-.', linewidth=2)

                    fig = plt.figure(3)
                    plot_tit = "OPTIMIZED "
                    plt.suptitle(plot_tit)

                    distO = numpy.sqrt((solO[:,0] - uO[:,0]) * (solO[:,0] - uO[:,0]) + (solO[:,1] - uO[:,1]) * (solO[:,1] - uO[:,1]))
                    distI = numpy.sqrt((solI[:,0] - uI[:,0]) * (solI[:,0] - uI[:,0]) + (solI[:,1] - uI[:,1]) * (solI[:,1] - uI[:,1]))

                    print(" END INTERVALS X", solI[0,0], solI[-1,0], uI[0,0], uI[-1,0], " ERROR ", distI[0])
                    print(" END INTERVALS Y", solI[0,1], solI[-1,1], uI[0,1], uI[-1,1], " ERROR ", distI[-1])


                    plt.plot(tO, solO[:,0] - uO[:,0], c = 'b'  , linestyle='-.', linewidth=3)
                    plt.plot(zO, solO[:,0] - uO[:,0], c = 'c'  , linestyle='-', linewidth=3)
                    plt.plot(tO, solO[:,1] - uO[:,1], c = 'orange'  , linestyle='-.', linewidth=3)
                    plt.plot(zO, solO[:,1] - uO[:,1], c = 'r'  , linestyle='-', linewidth=3)
                    plt.plot(tO, distO, c = 'g'  , linestyle='--', linewidth=3)

                    plt.scatter(tO[0], 0, c = 'b',  linewidth=4)
                    plt.scatter(tO[-1], 0, c = 'b',  linewidth=4)
                    plt.scatter(zO[0], 0, c = 'r')
                    plt.scatter(zO[-1], 0, c = 'r')


                    fig = plt.figure(4)
                    plot_tit = "INTERPOL "
                    plt.suptitle(plot_tit)
                    plt.plot(tI, solI[:,0] - uI[:,0], c = 'b'  , linestyle='-.', linewidth=3)
                    plt.plot(zI, solI[:,0] - uI[:,0], c = 'c'  , linestyle='-', linewidth=3)
                    plt.plot(tI, solI[:,1] - uI[:,1], c = 'orange'  , linestyle='-.', linewidth=3)
                    plt.plot(zI, solI[:,1] - uI[:,1], c = 'r'  , linestyle='-', linewidth=3)

                    plt.plot(tI, distI, c = 'g'  , linestyle='--', linewidth=3)
                    plt.scatter(tI[0], 0, c = 'b',  linewidth=4)
                    plt.scatter(tI[-1], 0, c = 'b',  linewidth=4)
                    plt.scatter(zI[0], 0, c = 'r')
                    plt.scatter(zI[-1], 0, c = 'r')



        print("------------------------------------------------------------------------")
        print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pU," ----------------")
        print("------------------------------------------------------------------------\n")

        print("\n \t|| x_p^* - alpha t_q || ==>  expect 2p =", 2 * pX,"\n\n")
        print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
        print("------------------------------------------------------------------------\n")
        for r in range(nR + 1):
            ne1 = pow(2, r) * ne
            if r == 0:
                print (ne1,"\t%1.3e"%disp_O[r],"         |","%1.3e"%numpy.sqrt(disp_O[r]))
            else:
                a = numpy.log10(           disp_O[r-1] /            disp_O[r]) / numpy.log10(2.0)
                b = numpy.log10(numpy.sqrt(disp_O[r-1])/ numpy.sqrt(disp_O[r])) / numpy.log10(2.0)
                print (ne1,"\t%1.3e"%disp_O[r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(disp_O[r]),"  %1.2f" %b)
        print("____________________________________________________________________\n")
        print("\n \t|| I_p^* - alpha t_q || ==>  expect max(p,q) + 1 =", max(pX, pU) + 1,"\n\n")
        print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
        print("------------------------------------------------------------------------\n")
        for r in range(nR + 1):
            ne1 = pow(2, r) * ne
            if r == 0:
                print (ne1,"\t%1.3e"%disp_I[r],"          |","%1.3e"%numpy.sqrt(disp_I[r]))
            else:
                a = numpy.log10(           disp_I[r-1] /            disp_I[r])  / numpy.log10(2.0)
                b = numpy.log10(numpy.sqrt(disp_I[r-1])/ numpy.sqrt(disp_I[r])) / numpy.log10(2.0)
                print (ne1,"\t%1.3e"%disp_I[r],"  %1.2f"%a, "  | %1.3e"%numpy.sqrt(disp_I[r]),"  %1.2f" %b)
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
    if ( curve == 0): I = [0, numpy.pi * 0.5]
    else:             I = [0.25, 1]

    TestDistanceFunctionOptimization.testDistanceFunction(degX, degT, elmts, refine, curve, I, showPlots)
