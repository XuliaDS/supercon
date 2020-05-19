import numpy
import math
import unittest
import pylab
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import Globals.configPaths

import Optimization.DistanceFunction.DistanceFunctionOptimization
from Optimization.DistanceFunction import OptimizationMaker

import Writers.VTKMeshWriter

import Geometry.FrechetDistance
import Geometry.ProjectionDistance
import Discretization.MasterElement
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

import Discretization.MasterElement.MasterElement1D

import Discretization.MasterElement.MasterElementMaker

import polynomial
import quadratures

import parula
def orientation(x, y):
    det = x[0] * y[1] - y[0] * x[1]      # determinant
    if (det > 0): return 1.0
    return -1.0

def signed_norm(dim, sizex, x, dx):
    if dim == 1:
        if sizex == 1: return x
        else: return x[:,0]
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

def mydistance (dim, n, x, y):
    return xnorm(dim, n, x - y)

def xnorm (dim, n, x):
    if dim == 1:    return abs(x)
    else:
        if n == 1:
            return numpy.sqrt(x[0] ** 2 + x[1]**2)
        z = numpy.zeros(n)
        for i in range(n):
            z[i] = numpy.sqrt(x[i,0] ** 2 + x[i,1] ** 2)
        return z

def convergence_IO(nR, ne, value, pX, pT, title):
    print("____________________________________________________________________\n")
    print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pT," ----------------")
    print("____________________________________________________________________\n")
    print("\n",title,"\n\n")
    print ("N\t E(x,t)\t    ORDER    sqrt(E)    ORDER")
    print("------------------------------------------------------------------------\n")
    for r in range(nR + 1):
        ne1 = pow(2, r) * ne
        if r == 0:
            print (ne1,"\t%1.3e"%value[r],"         |","%1.3e"%numpy.sqrt(value[r]))
        else:
            a = numpy.log10(           value[r-1] /            value[r])  / numpy.log10(2.0)
            b = numpy.log10(numpy.sqrt(value[r-1])/ numpy.sqrt(value[r])) / numpy.log10(2.0)
            print (ne1,"\t%1.3e"%value[r],"  %1.1f"%a, "  | %1.3e"%numpy.sqrt(value[r]),"  %1.1f" %b)
    print("____________________________________________________________________\n")




def newton_root (guess, eL, eR, polyX, polyT, f):
    tol   = 1.e-15
    itMAX = 20
    r     = 1.0 #0.5 * (eR - eL)
    #eM    = 0.5 * (eR + eL)
    zp   = guess
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

def fig_leg(root, n):
    ax        = pylab.gca()
    figlegend = pylab.figure(figsize=(3,2))
    # produce a legend for the objects in the other figure
    ax.legend(ncol = n)
    pylab.figlegend(*ax.get_legend_handles_labels(), ncol = 5,  loc = 'center', frameon = False)
    figlegend.show()
    os = root + 'legend.eps'
    figlegend.savefig(os, bbox_inches='tight', pad_inches=0)

def fig_header(count, s):
    fig = plt.figure(count)
    fig = plt.rc('text', usetex=True)
    fig = plt.rc('font', family='serif')
    fig = plt.rc('font',size=s)
    fig = plt.tick_params(which='both',      # both major and minor ticks are affected
                          top=False)
    fig = plt.figure(frameon=False)
    return count + 1
def change_mesh_nodes (mesh, ngp):
    newMasterElementX      = mesh.theMasterElementMakerX.createMasterElement(mesh.theDegreeX, ngp - 1)
    newMasterElementU      = mesh.theMasterElementMakerU.createMasterElement(mesh.theDegreeU, ngp - 1)
    mesh.theMasterElementX = newMasterElementX
    mesh.theMasterElementU = newMasterElementU
    w   = mesh.theMasterElementX.theGaussWeights
    z   = mesh.theMasterElementX.theGaussPoints
    return z, w

def evaluate_at_points(mesh, z, eID):
    shapeX, shapeDX = Discretization.MasterElement.MasterElement.orthopolyEdges(
                            mesh.theMasterElementX.theNodes[:,0],
                            z, mesh.theDegreeX)
    shapeT, shapeDT = Discretization.MasterElement.MasterElement.orthopolyEdges(
                            mesh.theMasterElementU.theNodes[:,0],
                            z, mesh.theDegreeU)

    X  = numpy.dot(shapeX,  mesh.theNodes                  [mesh.theElementsX[eID,:],:])
    T  = numpy.dot(shapeT,  mesh.theParametricNodes        [mesh.theElementsU[eID,:],:])
    DX = numpy.einsum('li,klj->kij',mesh.theNodes          [mesh.theElementsX[eID,:],:],shapeDX)
    DT = numpy.einsum('li,klj->kij',mesh.theParametricNodes[mesh.theElementsU[eID,:],:],shapeDT)
    return X, DX, T, DT


def fig_params(xax, yax, x, y, x2, y2, subdiv, xl):

    xt = [min(min(x), min(x2)), max(max(x), max(x2))]
    yt = [min(min(y), min(y2)), max(max(y), max(y2))]
    dt = 1.0 / (subdiv - 1)
    v2 = numpy.zeros(subdiv)
    v1 = numpy.zeros(subdiv)

    for j in range(subdiv):
        v1[j] = xt[0] + j * dt * (xt[1] - xt[0])
        v2[j] = yt[0] + j * dt * (yt[1] - yt[0])

    plt.xticks(v1)
    plt.yticks(v2)
    plt.xlabel(xl)
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter(xax))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter(yax))
    return

def fig_header(count, s):
    fig = plt.figure(count)
    #fig = plt.rc('text', usetex=True)
    fig = plt.rc('font', family='serif')
    fig = plt.rc('font',size=s)
    fig = plt.tick_params(which='both',      # both major and minor ticks are affected
                            top=False)
    return count + 1


class mypalette:
    def __init__(self,nC, showColors=False):
        nC = 40
        self.totColors = nC
        self.cmap      = parula.parula_map(numpy.linspace(0,1, nC))
        #cmap     = plt.cm.Paired(numpy.linspace(0,1, n_col))
        #cmap     = plt.cm.tab20(numpy.linspace(0,1, n_col))
        if showColors == True:
            for i in range(n_col):
                x = [0,1]
                y = [i,i]
                plt.plot(x,y, c = cmap[i], label = 'i =' + str(i))
            plt.show()
            #self.ls = numpy.empty((5))

        self.ls = [ (0,()) ,
                    (0, (3, 1, 1, 1)),
                    (0,(5,1)),
                    (0,(1,1)),
                    (0,(3,1,1,1,1,1))]
        self.c = [self.cmap[0] ,self.cmap[4],self.cmap[8],
                  self.cmap[12],self.cmap[16]]
        self.lw = [2,2,2,2,2]

        self.ax = self.cmap[1]



    def showColorPalette(self):
        j = 0
        for i in range(self.totColors):
            x = [0,1]
            y = [i,i]
            if (j == 5): j = 0
            plt.plot(x,y, ls = xp.ls[j], linewidth= xp.lw[j], c = self.cmap[i],  label = 'i =' + str(i))
            j += 1
        plt.show()


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

    def getGeometry2D(c, a, b):
        if   c == 4: return Curve2DPol.Curve2DExp (a, b)
        elif c == 1: return Curve2DPol.Curve2DSine(a, b)
        elif c == 11:return Curve2DPol.Curve2DSineSine(a, b)
        elif c == 2: return Curve2DPol.Curve2DPol2(a, b)
        elif c == 3: return Curve2DPol.Curve2DPol3(a, b)
        elif c == 6: return Curve2DPol.Curve2DPol6(a, b)

        elif c == 0 or c == 10: return Circle.Circle (1.0, a, b) #Curve2DPol.Curve2DCircle(a,b)
        elif c == 5: return CirclePolynomial.CirclePolynomial(1, 2)
        elif c == 8: return Curve2DPol.Curve2DHypCircle(a,b)
        elif c == 9: return Curve2DPol.Curve2DsinExp(a,b)
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





        #xp.showColorPalette()

    class mesh_info:
        def __init__(self, dim,  mesh, parametrization, Q, type):
            self.z, self.w = quadratures.qType(Q, type)
            self.n    = mesh.theNOfElements
            self.x    = numpy.empty((self.n, Q, dim  ))
            self.t    = numpy.empty((self.n, Q, 1    ))
            self.dt   = numpy.empty((self.n, Q, 1  ,1))
            self.dx   = numpy.empty((self.n, Q, dim,1))
            self.err  = numpy.empty((self.n, Q       ))
            self.derr = numpy.empty((self.n, Q       ))
            self.aot  = numpy.empty((self.n, Q, dim  ))
            self.daot = numpy.empty((self.n, Q, dim,1))
            self.Q    = Q
            for i in range (self.n):
                self.x[i], self.dx[i], self.t[i], self.dt[i] = evaluate_at_points(mesh, self.z, i)
                self.aot [i] = parametrization.value(self.t[i])
                self.daot[i] = parametrization.tangent(self.t[i])
                self.err [i] = signed_norm(dim, Q, self.aot[i] - self.x[i] , self.daot[i])
                self.derr[i]  = xnorm(dim, Q, self.daot[i]*self.dt[i] - self.dx[i])
                #hess = parametrization.hessian(self.t[i])
                #self.derr[i] = signed_norm(dim, Q, self.daot[i] * self.dt[i] - self.dx[i] , hess)


    @staticmethod
    def testDistanceFunction(dim, pX, pT, ne, nR, curve, I, showPlots, mesh_IO):

        relocateX = False
        fixU      = False
        callFix   = True
        method    = 'Newton'
        tolDistanceCalculation = 1.e-8
        tol = 1.e-8

        disparity             = numpy.zeros([nR + 1])

        gp                    = 100
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        if (dim == 1): parametrization = TestDistanceFunctionOptimization.getGeometry1D(curve, I[0], I[1])
        else:          parametrization = TestDistanceFunctionOptimization.getGeometry2D(curve, I[0], I[1])

        FS = 20
        ea       = numpy.zeros(2)
        dea      = numpy.zeros(2)
        pltInfo  = '    pX = '+str(pX)+' pT = '+str(pT)

        if dim == 1: pE = pX + pT
        else:        pE =  2 * pX
        pS = 4 * pX
        pD = pE * 2
        n_poly_tests   = 1
        reconstruction = ['Modal', 'Modal', 'Modal', 'Nodal']
        #poly_type      = [polynomial.eChebiFirst, polynomial.eLegendre]
        poly_type      = ['Legendre',               'Chebyshev',       'Legendre',               'Monomial']
        quadrature     = ['Gauss-Lobatto-Legendre', 'Gauss-Chebyshev','Gauss-Legendre' , 'Gauss-Lobatto-Legendre']
        plt_tits = []
        # Store GPs and shift them for multiple elements


        f_mode = 0
        f_poly = n_poly_tests + 1
        f_deco = n_poly_tests + 2
        fcount = n_poly_tests + 3



        xp = mypalette(20)
        #xp.showColorPalette()

        disparity_repro = numpy.zeros((n_poly_tests, nR + 1))
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
                fixU      = fixU )

            mesh, meshI = optimizer.run()
            if meshIO == -1:
                print(' Attention: we are showing results for initial mesh !!! ')
                mesh = meshI
                if dim == 1: pE = max(pX, pT) + 1
                else:        pE = pX + 1

            n               = mesh.theNOfElements
            disf,proje,norm = TestDistanceFunctionOptimization.getMeshDistances(
                                    mesh,parametrization,frechetFunctionName,
                                    tolDistanceCalculation, gp - 1)

            disparity[ref]  = disf * disf * 0.5

            zex   = numpy.empty((n *  gp, 1))
            # Element Boundaries
            eBD   = numpy.zeros(n + 1)
            x     = numpy.empty((n * gp,dim))
            t     = numpy.empty((n * gp,dim))
            aot   = numpy.empty((n * gp,dim))
            errC  = numpy.empty((n * gp,dim)) # error components
            derrC = numpy.empty((n * gp,dim)) # error components

            errT  = numpy.empty((n * gp    )) # total error

            x_poly     = numpy.empty((n_poly_tests,n * gp,dim))
            aot_poly   = numpy.empty((n_poly_tests,n * gp,dim))


            errC_poly  = numpy.empty((n_poly_tests,n * gp,dim))
            errT_poly  = numpy.empty((n_poly_tests,n * gp    ))
            errTT_poly = numpy.empty((n_poly_tests,n * gp    ))
            derrC_poly = numpy.empty((n_poly_tests,n * gp,dim)) # error components

            e_exp_by_mode      = numpy.zeros((n_poly_tests, pE + 1, n * gp))
            e_exp_by_mode_dim  = numpy.zeros((n_poly_tests, pE + 1, n * gp, dim))
            aot_exp_by_mode    = numpy.zeros((n_poly_tests, pS + 1, n * gp, dim))
            de_exp_by_mode_dim = numpy.zeros((n_poly_tests, pD + 1, n * gp, dim))

            x_poly_exp_by_mode = numpy.empty((n_poly_tests, pX + 1, n * gp, dim))

            for i in range(n + 1):
                eBD[i] = parametrization.theT0 + h * i

            dumb = numpy.zeros([1,1])
            for pt in range(n_poly_tests):
                disparity_repro[pt, ref] = 0.0
                eQT = quadrature[pt]
                if (poly_type[pt] == 'Chebyshev'): eQT = 'Gauss-Chebyshev'
                eQT      = 'Gauss-Lobatto-Legendre'
                m_pe     = TestDistanceFunctionOptimization.mesh_info(dim, mesh, parametrization, pE + 1, eQT)
                m_ps     = TestDistanceFunctionOptimization.mesh_info(dim, mesh, parametrization, pS + 1, eQT)
                m_px     = TestDistanceFunctionOptimization.mesh_info(dim, mesh, parametrization, pX + 1, eQT)
                m_sample = TestDistanceFunctionOptimization.mesh_info(dim, mesh, parametrization,     gp, eQT)
                m_pt     = TestDistanceFunctionOptimization.mesh_info(dim, mesh, parametrization, pT + 1, eQT)

                testCriticalCondition = True
                if testCriticalCondition == True:
                    for e in range(n):

                        # check critical condition for test space Xp
                        #get shape functions
                        shapeX, shapeDX = Discretization.MasterElement.MasterElement.orthopolyEdges(
                                                mesh.theMasterElementX.theNodes[:,0],
                                                m_sample.z, mesh.theDegreeX)
                        shapeT, shapeDT = Discretization.MasterElement.MasterElement.orthopolyEdges(
                                                mesh.theMasterElementU.theNodes[:,0],
                                                m_sample.z, mesh.theDegreeU)

                        dis = m_sample.x[e] - m_sample.aot[e]
                        if dim == 2:
                            sig = xnorm(2, gp, m_sample.dx[e])  # || x' || 
                        else: sig = abs(m_sample.dx[e])
                        zerosum = 0.0
                        for k in range(pX + 1):
                        #    plt.plot(m_sample.z, shapeX[:,k], label='N-' + str(k))
                            for d in range(dim):
                                zerotest = 0.0
                                for j in range(gp):
                                    aux  = dis[j,d]           *  shapeX[j,k]
                                    aux2 = m_sample.dx[e,j,d] * shapeDX[j,k]
                                    #print(sig[j], ' and ',numpy.dot(dis[j],shapeX[j,k]) )
                                    term_1 = sig[j] * aux
                                    term_2 =    0.5 * numpy.dot(dis[j], dis[j]) * aux2 / sig[j]
                                    zerotest += m_sample.w[j] * (term_1 - term_2)
                                print('X test ', k, ' zero = ', zerotest)
                            zerosum += zerotest
                            print(' TOTAL SUM ', zerosum)
                        #plt.show()
                        tsum = 0
                        for k in range(pT + 1):
                            sum = 0.0
                    #        plt.plot(m_sample.z, shapeT[:,k], label='N-' + str(k))
                            for j in range(gp):
                                term_1 = numpy.dot(dis[j], m_sample.daot[e,j] * m_sample.dt[e,j]) * shapeT[j,k]
                                sum   += m_sample.w[j] * term_1 * sig[j]
                            print('T test ', k, ' zero = ', sum)
                            tsum += sum
                        print(' TOTAL SUM ', tsum )
                    #    plt.show()
                        quit()

                for e in range(n):

                    # velocity ERROR
                    sig = xnorm(dim, gp, m_sample.dx[e])
                    edx = numpy.zeros((gp,dim))
                    for k in range(gp):
                        for d in range(dim):
                            edx[k,d] = (m_sample.x[e,k,d] - m_sample.aot[e,k,d]) * sig[k]

                    '''    plt.subplot(2,1,1)
                    plt.plot(m_sample.z, edx[:,0], c = xp.c[0], label='edx 0')
                    plt.plot(m_sample.z, edx[:,1], c = xp.c[1], label='edx 1')
                    plt.plot(m_sample.z, m_sample.x[e,:,0] - m_sample.aot[e,:,0], c = xp.c[2], label ='e0')
                    plt.plot(m_sample.z, m_sample.x[e,:,1] - m_sample.aot[e,:,1], c = xp.c[3], label = 'e1')
                    plt.legend()
                    plt.subplot(2,1,2)
                    plt.plot(m_sample.z, sig, c = xp.c[0])
                    plt.show()'''




                    sumXA   = 0.0
                    polyX     = polynomial.polynomial(dim, pX, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, m_sample.x[e])
                    polyA     = polynomial.polynomial(dim, pS, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, m_sample.aot[e])
                    polyE_dim = polynomial.polynomial(dim, pE, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, m_sample.aot[e] - m_sample.x[e])
                    polyE_sig = polynomial.polynomial(1,   pE, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, m_sample.err[e])

                    vd        = m_sample.daot[e] * m_sample.dt[e] - m_sample.dx[e]

                    polyDeriv = polynomial.polynomial(dim, pD, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, vd[:,:,0])
                    # Now compute signed error from 2D error
                    poly_error_sigdim = signed_norm(dim, gp, polyE_dim.value, m_sample.dx[e])
                    for j in range(gp):

                        zex       [   gp * e + j] = 0.5 * ( (eBD[e + 1] - eBD[e]) * m_sample.z[j] + eBD[e + 1] + eBD[e] )
                        x         [   gp * e + j] = m_sample.x[e,j]
                        t         [   gp * e + j] = m_sample.t[e,j]
                        aot       [   gp * e + j] = m_sample.aot[e,j]
                        errC      [   gp * e + j] = m_sample.aot[e,j] - m_sample.x[e,j]
                        errT      [   gp * e + j] = m_sample.err[e,j]
                        derrC     [   gp * e + j] = vd [j,:,0]

                        x_poly    [pt,gp * e + j] = polyX.value[j]
                        aot_poly  [pt,gp * e + j] = polyA.value[j]
                        errC_poly [pt,gp * e + j] = polyE_dim.value[j]
                        errT_poly [pt,gp * e + j] = polyE_sig.value[j]
                        derrC_poly[pt,gp * e + j] = polyDeriv.value[j]
                        errTT_poly[pt,gp * e + j] = poly_error_sigdim[j]
                        derx    = xnorm(dim, 1,m_sample.dx[e,j])
                        err_int = abs(m_sample.err[e,j])
                        wf      = 1.0
                        if (eQT == quadratures.eGC): wf = numpy.sqrt(1.0 - m_sample.z[j] * m_sample.z[j])
                        sumXA += err_int * err_int * derx * m_sample.w[j] * wf

                    disparity_repro[pt,ref] += sumXA * 0.5
                    if e == 0: plt_tits.append(polyX.getType())

                    dist = mydistance(dim, gp,  m_sample.x[e], polyX.value)
                    if (dist.sum(axis=0)> 1.e-12):
                        print(' !!!! Error in ', polyX.getType(), ' representation = ', ds1)

                    # PLOT MODES IN LOGSCALE FOR ERROR, MESH AND CURVE
                    if showPlots > 0 and ref == 0:
                        nsb = 4
                        if e == 0:
                            fig_header(f_mode + pt, FS)
                            #plt.suptitle(polyX.getType() + ' Coefficients')

                        for j in range (nsb):
                            plt.subplot(nsb / 2,nsb / 2,j + 1)
                            if j == 0:
                                poly = polyE_sig
                                if ( e == 0 ):
                                    plt.title('Signed Error')
                            elif j == 1:
                                poly = polyE_dim
                                if ( e == 0 ):
                                    plt.title('Dimensional Error')
                            elif j == 2:
                                poly = polyX
                                if ( e == 0 ):
                                    plt.title('$X_p$ in modal form')
                            else:
                                poly = polyA
                                if ( e == 0 ):
                                    plt.title(r'$L^2(\alpha \circ t)$ using $n$ =  '+ str(pS))

                            size  = 2
                            if j >= 2: size = 1
                            i_nor = numpy.empty((poly.n - size))
                            c_nor = numpy.empty((poly.n - size))
                            i_spe = numpy.empty((size))
                            c_spe = numpy.empty((size))
                            ik     = 0
                            jk     = 0
                            last = poly.n - 1
                            if j >= 2: last = -1
                            for k in range(poly.n):
                                dk = xnorm(poly.dim, 1, poly.node[k])
                                if k == pX or k == last:
                                    i_spe[ik] = k
                                    c_spe[ik] = numpy.log10(dk)
                                    ik        += 1
                                else:
                                    i_nor[jk]  = k
                                    c_nor[jk]  = numpy.log10(dk)
                                    jk        += 1

                            if e == 0:
                                idx = numpy.append(i_nor, i_spe)
                                coe = numpy.append(c_nor, c_spe)
                                fig_params('%1d', '%1.1f', idx, coe, idx, coe, poly.n, 'Coefficients')
                                plt.xticks(idx)
                            plt.scatter(i_nor, c_nor, color = xp.c[0], marker='o', s = 25)
                            plt.scatter(i_spe, c_spe, color = xp.c[1], marker='*', s = 50)





                    # break expansion
                    if ref == -1 and e == 0:
                        fig_header(f_poly, FS)
                        plt.subplot(n_poly_tests,1, pt + 1)
                        plt.title(polyE_sig.getType() + ' Basis')
                        yzero = numpy.zeros(gp)
                        for j in range (pE + 1):
                            plt.plot(m_sample.z, basisE[:,j],  color = cmap[j], label =r'$P_' + str(j)+'$')
                        fig_params('%1.1f', '%1.1f',m_sample.z, [-1,1], m_sample.z, [-1,1], '$\\xi$')


                    basisE    = polyE_sig.getBasis(pE + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1
                    basisEdim = polyE_dim.getBasis(pE + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1
                    basisA    = polyA.getBasis    (pS + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1
                    basisX    = polyX.getBasis    (pX + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1

                    basisDE   = polyDeriv.getBasis(pD + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1

                    for k in range(gp):
                        for j in range(max(pD + 1, pS + 1)):
                            if j <= pS:
                                aot_exp_by_mode[pt, j, gp * e  + k] = polyA.node[j] * basisA[k,j]
                            if j <= pE:
                                e_exp_by_mode     [pt, j, gp * e  + k] = polyE_sig.node[j] * basisE[k,j]
                                e_exp_by_mode_dim [pt, j, gp * e  + k] = polyE_dim.node[j] * basisEdim[k,j]
                            if j <= pD:
                                de_exp_by_mode_dim[pt, j, gp * e  + k] = polyDeriv.node[j] * basisDE[k,j]

                            if j <= pX:
                                x_poly_exp_by_mode[pt, j, gp * e  + k] = polyX.node[j] * basisX[k,j]


            if ref != 0: continue

            zAXIS  = numpy.zeros(n + 1)
            yAXIS  = numpy.zeros(n + 1)
            ypAXIS = numpy.zeros(n * (pX + 1))
            yuAXIS = numpy.zeros(n * (pT + 1))
            tAXIS  = numpy.zeros(n + 1)
            xEP    = numpy.zeros([n + 1,dim])
            aEP    = numpy.zeros([n + 1,dim])

            for i in range(n):
                zAXIS[i] = zex  [i * gp][0]
                tAXIS[i] = t    [i * gp][0]
                xEP  [i] = x    [i * gp]
                aEP  [i] = aot[i * gp]
            zAXIS[n] = zex  [-1][0]
            tAXIS[n] = t    [-1][0]
            xEP[n]   = x    [-1]
            aEP[n]   = aot[-1]

            def myplot(x, y, mp, j, name, zAXIS= None, yAXIS = None, plot_axis = False):
                plt.plot(x, y, ls = mp.ls[j], linewidth= mp.lw[j], color = mp.c[j], label = name)
                if (plot_axis):
                    plt.plot   (zAXIS,yAXIS, ls = ':' ,lw = 0.5, color = mp.ax)
                    plt.scatter(zAXIS,yAXIS,  s = 10,  color = mp.ax)


            for pt in (range(n_poly_tests)):
                def break_expansion(dim, f, m, n, p, q):
                    f0 = numpy.zeros((m, dim))
                    f1 = numpy.zeros((m, dim))
                    f2 = numpy.zeros((m, dim))
                    for j in range(m):
                        for i in range(n):
                            if   i < p: f0[j] += f[i,j]
                            elif i < q: f1[j] += f[i,j]
                            else      : f2[j] += f[i,j]
                    return f0, f1, f2


                e0p,  ep2pm1,  e2p        = break_expansion(1  , e_exp_by_mode[pt]     , n * gp, pE + 1, pX + 1, pE)
                de0p, dep2pm1, de2p       = break_expansion(dim, e_exp_by_mode_dim[pt] , n * gp, pE + 1, pX + 1, pE)
                a0p,  ap2pm1,  a2p        = break_expansion(dim, aot_exp_by_mode [pt]  , n * gp, pS + 1, pX + 1, pE)
                der_0p, der_p2pm1, der_2p = break_expansion(dim, de_exp_by_mode_dim[pt], n * gp, pD + 1, pX + 1, pE - 1)
                plt.subplot(1,2,1)
                plt.plot(zex, derrC[:,0])
                plt.plot(zex, derrC_poly[0,:,0])
                plt.subplot(1,2,2)
                plt.plot(zex, derrC[:,0])
                plt.plot(zex, derrC_poly[0,:,0])
                plt.show()

                def show_error_plot(mycolor, zex, xa, ya, e, ep, p0, p1, p2, pE, pX, fcount, title):


                    lp0 = 'coeffs 0,..,' + str(pX)

                    if (pX == 2):
                        lp1 = 'coeff ' + str(pE - 1)
                    elif (pX == 3):
                        lp1 = 'coeffs '+ str(pX + 1) + ',' + str(pE - 1)
                    else:
                        lp1 = 'coeffs '+ str(pX + 1) + '...' + str(pE - 1)
                    lp2 = 'coeff '  + str(pE)

                    print(' fig count ', fcount)
                    fcount = fig_header(fcount, FS)
                    print(' f count ', fcount)
                    plt.suptitle(title)
                    plt.subplot(2,2,1)
                    plt.title('overlap')
                    myplot(zex, e,  mycolor, 0, '$e\circ t$ ')
                    myplot(zex, ep, mycolor, 1, '$e_{'  + str(pE) +'}$')
                    myplot(zex, p2, mycolor, 4, lp2, xa, ya, True)
                    myplot(zex, p1, mycolor, 3, lp1)
                    myplot(zex, p0, mycolor, 2, lp0)
                    plt.legend()
                    fig_params('%1.1f', '%1.1e',zex, e, zex,  ep, 2, '$\\xi$')

                    plt.subplot(2,2,2)
                    plt.title('0,..,p')
                    myplot(zex, p0, mycolor,  2,  lp0, xa, ya, True)
                    fig_params('%1.1f', '%1.1e',zex, p0, zex, p0, 2 , '$\\xi$')

                    plt.subplot(2,2,3)
                    plt.title('p,..,2p -1')
                    myplot(zex, p1, mycolor,  3,  lp1, xa, ya, True)
                    fig_params('%1.1f', '%1.1e',zex, p1, zex, p1, 2 , '$\\xi$')

                    plt.subplot(2,2,4)
                    plt.title('2p,')
                    myplot(zex, p2, mycolor,  4,  lp2, xa, ya, True)

                    fig_params('%1.1f', '%1.1e',zex, p2, zex, p2, 2 , '$\\xi$')

                    print(' leave with ', fcount)
                    return fcount


                def error_plot(mycolor, zex, xa, ya, e, ep, p0, p1, p2, pE, pX,  oname, fcount, showleg):


                    lp0 = 'coeffs 0,..,' + str(pX)

                    if (pX == 2):
                        lp1 = 'coeff ' + str(pE - 1)
                    elif (pX == 3):
                        lp1 = 'coeffs '+ str(pX + 1) + ',' + str(pE - 1)
                    else:
                        lp1 = 'coeffs '+ str(pX + 1) + '...' + str(pE - 1)
                    lp2 = 'coeff '  + str(pE)

                    fcount = fig_header(fcount, FS)
                    myplot(zex, e,  mycolor, 0, '$e\circ t$ ')
                    myplot(zex, ep, mycolor, 1, '$e_{'  + str(pE) +'}$')
                    myplot(zex, p2, mycolor, 4, lp2, xa, ya, True)
                    myplot(zex, p1, mycolor, 3, lp1)
                    myplot(zex, p0, mycolor, 2, lp0)
                    of = oname + 'Overlap.eps'
                    fig_params('%1.1f', '%1.1e',zex, e, zex,  ep, 2, '$\\xi$')
                    plt.savefig(of, bbox_inches='tight', pad_inches=0)
                    if (showleg == True):
                        fig_leg(oname + 'Overlap', 5)
                    plt.close()
                    fcount -= 1

                    showleg = False
                    of     = oname + '0p.eps'
                    fcount = fig_header(fcount, FS)
                    myplot(zex, p0, mycolor,  2,  lp0, xa, ya, True)
                    fig_params('%1.1f', '%1.1e',zex, p0, zex, p0, 2 , '$\\xi$')
                    plt.savefig(of, bbox_inches='tight', pad_inches=0)
                    if (showleg == True):
                        fig_leg(oname + '0p')
                    fcount -= 1
                    plt.close()


                    fcount = fig_header(fcount, FS)
                    myplot(zex, p1, mycolor,  3,  lp1, xa, ya, True)
                    fig_params('%1.1f', '%1.1e',zex, p1, zex, p1, 2 , '$\\xi$')
                    of = oname + 'p2p-1.eps'
                    plt.savefig(of, bbox_inches='tight', pad_inches=0)
                    if (showleg == True):
                        fig_leg(oname + 'p2p-1')
                    fcount -= 1
                    plt.close()

                    fcount = fig_header(fcount, FS)
                    myplot(zex, e , mycolor,  0, lp2, xa, ya, True)
                    myplot(zex, p2, mycolor,  4, lp2, xa, ya, True)

                    fig_params('%1.1f', '%1.1e',zex, p2, zex, e, 2 , '$\\xi$')

                    of = oname + '2p.eps'
                    plt.savefig(of, bbox_inches='tight', pad_inches=0)
                    if (showleg == True):
                        fig_leg(oname + '2p')
                    fcount -= 1
                    plt.close()
                    return fcount

                #plt.suptitle(r'$e \circ t = sgn\cdot || X - \alpha \circ t || $ ' + plt_tits[pt]+  ' ' + pltInfo)
                print ('total error index ', fcount)
                fcount = error_plot(xp, zex, zAXIS, yAXIS, errT, errT_poly[pt], \
                            e0p, ep2pm1, e2p, pE, pX, 'tot_err', fcount, False)

                print ('1st comp error index ', fcount)
                fcount = error_plot(xp, zex, zAXIS, yAXIS, errC[:,0], errC_poly[pt,:,0], \
                            de0p[:,0], dep2pm1[:,0], de2p[:,0], pE, pX,  'first_err', fcount, False)

                print ('1st comp alpha index ', fcount)
                fcount = error_plot(xp, zex, zAXIS, yAXIS, aot[:,0], aot_poly[pt,:,0], \
                            a0p[:,0], ap2pm1[:,0], a2p[:,0], pE, pX,  'first_alpha', fcount, False)


                print ('1stcomp derivative error index ', fcount)

                fcount = error_plot(xp, zex, zAXIS, yAXIS, derrC[:,0], derrC_poly[pt,:,0], \
                            der_0p[:,0], der_p2pm1[:,0], der_2p[:,0], pE, pX, 'der_first_err',fcount, True)


                print ('2nd comp error index ', fcount)

                fcount = error_plot(xp, zex, zAXIS, yAXIS, errC[:,1], errC_poly[pt,:,1], \
                            de0p[:,1], dep2pm1[:,1], de2p[:,1], pE, pX, 'second_err',fcount, True)
                print ('2nd comp alpha index ', fcount)

                fcount = error_plot(xp, zex, zAXIS, yAXIS, aot[:,1], aot_poly[pt,:,1], \
                            a0p[:,1], ap2pm1[:,1], a2p[:,1], pE, pX,  'second_alpha', fcount, True)


                print ('2nd comp derivative error index ', fcount)

                fcount = error_plot(xp, zex, zAXIS, yAXIS, derrC[:,1], derrC_poly[pt,:,1], \
                            der_0p[:,1], der_p2pm1[:,1], der_2p[:,1], pE, pX, 'der_second_err',fcount, True)
                fcount += 1

                #plt.suptitle(r'$e \circ t = sgn\cdot || X - \alpha \circ t || $ ' + plt_tits[pt]+  ' ' + pltInfo)
                print ('total error index ', fcount)
                fcount = show_error_plot(xp, zex, zAXIS, yAXIS, errT, errT_poly[pt], \
                            e0p, ep2pm1, e2p, pE, pX, fcount, 'error total ')

                print ('1st comp error index ', fcount)
                fcount = show_error_plot(xp, zex, zAXIS, yAXIS, errC[:,0], errC_poly[pt,:,0], \
                            de0p[:,0], dep2pm1[:,0], de2p[:,0], pE, pX, fcount, 'error 1st comp')

        #        print ('1st comp alpha index ', fcount)
        #        fcount = show_error_plot(xp, zex, zAXIS, yAXIS, aot[:,0], aot_poly[pt,:,0], \
        #                    a0p[:,0], ap2pm1[:,0], a2p[:,0], pE, pX, fcount,  'alpha 1st comp')


                print ('1stcomp derivative error index ', fcount)

                fcount = show_error_plot(xp, zex, zAXIS, yAXIS, derrC[:,0], derrC_poly[pt,:,0], \
                            der_0p[:,0], der_p2pm1[:,0], der_2p[:,0], pE, pX, fcount,  'der error 1st comp')


                print ('2nd comp error index ', fcount)

                fcount = show_error_plot(xp, zex, zAXIS, yAXIS, errC[:,1], errC_poly[pt,:,1], \
                            de0p[:,1], dep2pm1[:,1], de2p[:,1], pE, pX, fcount,  'error 2nd comp')

        #        print ('2nd comp alpha index ', fcount)

        #       fcount = show_error_plot(xp, zex, zAXIS, yAXIS, aot[:,1], aot_poly[pt,:,1], \
        #                    a0p[:,1], ap2pm1[:,1], a2p[:,1], pE, pX, fcount,  'alpha 2nd comp')

                print ('2nd comp derivative error index ', fcount)

                fcount = show_error_plot(xp, zex, zAXIS, yAXIS, derrC[:,1], derrC_poly[pt,:,1], \
                            der_0p[:,1], der_p2pm1[:,1], der_2p[:,1], pE, pX,fcount,  'der error 2nd comp')
                fcount += 1



        for pt in range(n_poly_tests):
            mat = pE + 2
            fcount = fig_header(fcount, FS)
            plt.suptitle(plt_tits[pt] + ' error decomposition')
            if (mat % 2 != 0):
                half = int( (mat + 1) / 2)
                col  = half
                row  = mat / half - 1
            else:
                half = int(mat / 2)
                col  = half
                row  = mat / half
            for j in range (mat):
                plt.subplot(row, col , j + 1)
                if j == pE + 1:
                    plt.title('All')
                    fun = errT_poly[pt]
                else:
                    plt.title('$P_' + str(j)+'$' )
                    fun = e_exp_by_mode[pt,j]
                plt.plot(zex, fun, color = xp.c[0] )
                funn = numpy.zeros(n * gp + 1)
                funn[0] = 0.0
                #for i in range(gp + 1):
            #        funn[i + 1] = fun[i]
            #    fig_params('%1.1f', '%1.1e',zex, funn, zex, funn)
                auxY = numpy.zeros( 2 * n )
                auxX = numpy.zeros( 2 * n )
                for e in range(n):
                    auxY[2 * e    ] = fun[gp * e ]
                    auxY[2 * e + 1] = fun[gp * e  + gp - 1]
                    auxX[2 * e    ] = zex[gp * e ]
                    auxX[2 * e + 1] = zex[gp * e  + gp - 1]
                plt.scatter(auxX, auxY  , color = xp.c[0], s = 15)
                plt.plot   (zAXIS,yAXIS, linestyle = ':'   ,linewidth= 0.5, color = xp.ax)
                plt.scatter(zAXIS,yAXIS,   color = xp.ax, s = 10)

        convergence_IO(nR, ne, disparity, pX, pT, ' DISPARITY ORIGINAL MESH')
        for pt in range(n_poly_tests):
            name='Disparity using a ' + reconstruction[pt] + ' basis and ' + poly_type[pt] + \
                ' polynomials. Solving with ' + quadrature[pt]+ ' rules'
            convergence_IO(nR, ne, disparity_repro[pt], pX, pT, name)


        if (showPlots >= 1):
        #    for j in range(fcount):
        #        plt.figure(j)
        #        plt.savefig('overleaf/'+str(j)+'.eps', format='eps', dpi=1000)
            plt.show()




if __name__ == '__main__':

    argc = len(sys.argv)

    if argc < 8:
        print (" I NEED dimension + degree x  + degree t + initial elements + refinements + cure type + show plots")
        print(sys.argv)
        quit(1)
    meshIO    = 0
    dim       = int(sys.argv[1])  # number of elements
    degX      = int(sys.argv[2])  # number of elements
    degT      = int(sys.argv[3])  # number of elements
    elmts     = int(sys.argv[4])  # number of elements
    refine    = int(sys.argv[5])  # number of elements
    curve     = int(sys.argv[6])  # number of elements
    showPlots = int(sys.argv[7])  # number of elements
    if argc == 9: meshIO = int(sys.argv[8])
    print(' SPACE DIMENSIONS ',dim)
    I = [0,1]
    if dim == 2:
        if ( curve == 0 ):
            I = [0,numpy.pi]
            print(" SOLVING alpha = (cos(x), sin(x)) x in [0, pi]")
        elif ( curve == 10):
            I = [0,2 * numpy.pi]
            print(" SOLVING alpha = (cos(x), sin(x)) x in [0, 2pi]")
    else:
        if   (curve ==  0 ):
            I = [0, numpy.pi]
            print(" SOLVING COS(x) x in [0, pi]")
        elif   ( curve == 7):
            I = [-numpy.pi * 0.5, numpy.pi * 0.5]
            print(" SOLVING sin(x) in [-pi/2, pi/2]")
        elif (curve == 10):
            I = [0, 2.0 * numpy.pi]
            print(" SOLVING COS(x) x in [0, 2pi]")
        elif (curve == 5):
            I = [1, 2]
            print(" SOLVING a poly deg 5 ")
    TestDistanceFunctionOptimization.testDistanceFunction(dim, degX, degT, elmts, refine, curve, I, showPlots, meshIO)
