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
import errAn


import vector_functions as vf

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


    class mesh_info:
        def __init__(self, dim,  mesh, parametrization, Q, type):
            self.z, self.w = quadratures.qType(Q, type)
            self.n    = mesh.theNOfElements
            self.x    = numpy.empty((self.n, Q, dim  ))
            self.t    = numpy.empty((self.n, Q, 1    ))
            self.dt   = numpy.empty((self.n, Q, 1, 1  ))
            self.dx   = numpy.empty((self.n, Q, dim,1))
            self.err  = numpy.empty((self.n, Q       ))
            self.aot  = numpy.empty((self.n, Q, dim  ))
            self.daot = numpy.empty((self.n, Q, dim,1))
            self.Q    = Q
            for i in range (self.n):
                self.x[i], self.dx[i], self.t[i], self.dt[i] = evaluate_at_points(mesh, self.z, i)
                self.aot[i]     = parametrization.value(self.t[i])
                self.daot[i]    = parametrization.tangent(self.t[i])
                self.daot[i,:] *= self.dt[i]
                self.err[i]     = vf.signed_norm(dim, Q, self.aot[i] - self.x[i] , self.dx[i])


    @staticmethod
    def testDistanceFunction(dim, pX, pT, ne, nR, curve, I, showPlots, mesh_IO):

        relocateX = False
        fixU      = False
        callFix   = True
        method    = 'Newton'
        showLeg   = False
        if showPlots == 2:
            visType = 0
        elif showPlots == 1:
            showLeg = True
            visType = 1
        tolDistanceCalculation = 1.e-12
        tol = 1.e-12

        disparity             = numpy.zeros([nR + 1])

        gp                    = 100
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        if (dim == 1): parametrization = TestDistanceFunctionOptimization.getGeometry1D(curve, I[0], I[1])
        else:          parametrization = TestDistanceFunctionOptimization.getGeometry2D(curve, I[0], I[1])

        ea       = numpy.zeros(2)
        dea      = numpy.zeros(2)
        pltInfo  = '    pX = '+str(pX)+' pT = '+str(pT)

        if dim == 1: pE = pX + pT
        else:        pE =  2 * pX
        pS = 4 * pX
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

        xp = errAn.mypalette(20)
    #    xp.showColorPalette()



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
            errT  = numpy.empty((n * gp    )) # total error

            x_poly     = numpy.empty((n_poly_tests,n * gp,dim))
            aot_poly   = numpy.empty((n_poly_tests,n * gp,dim))

            errC_poly  = numpy.empty((n_poly_tests,n * gp,dim))
            errT_poly  = numpy.empty((n_poly_tests,n * gp    ))
            errTT_poly = numpy.empty((n_poly_tests,n * gp    ))

            e_exp_by_mode      = numpy.zeros((n_poly_tests, pE + 1, n * gp))
            e_exp_by_mode_dim  = numpy.zeros((n_poly_tests, pE + 1, n * gp, dim))
            aot_exp_by_mode    = numpy.zeros((n_poly_tests, pS + 1, n * gp, dim))
            x_poly_exp_by_mode = numpy.empty((n_poly_tests, pX + 1, n * gp, dim))


            for i in range(n + 1):
                eBD[i] = parametrization.theT0 + h * i

            dumb = numpy.zeros([1,1])
            for pt in range(n_poly_tests):
                disparity_repro[pt, ref] = 0.0
                eQT = quadrature[pt]
                if (poly_type[pt] == 'Chebyshev'): eQT = 'Gauss-Chebyshev'
                m_pe     = TestDistanceFunctionOptimization.mesh_info(dim, mesh, parametrization, pE + 1, eQT)
                m_ps     = TestDistanceFunctionOptimization.mesh_info(dim, mesh, parametrization, pS + 1, eQT)
                m_px     = TestDistanceFunctionOptimization.mesh_info(dim, mesh, parametrization, pX + 1, eQT)
                m_sample = TestDistanceFunctionOptimization.mesh_info(dim, mesh, parametrization,     gp, eQT)

                for e in range(n):
                    polyX     = polynomial.polynomial(dim, pX, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, m_sample.x[e])
                    polyA     = polynomial.polynomial(dim, pS, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, m_sample.aot[e])
                    dotdx = numpy.zeros(gp)
                    e_dim = numpy.zeros((gp, dim))
                    e_sig = numpy.zeros(gp)

                    example_l2_accuracy = False
                    if (example_l2_accuracy == True):
                        quartic = numpy.zeros(gp)
                        for j in range(gp): quartic[j] = numpy.power(m_sample.z[j], 4) + m_sample.z[j]**2
                        cubic_poly  = polynomial.polynomial(1, 3, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                           m_sample.w, eQT, quartic )
                        err_quartic = polynomial.polynomial(1, 4, reconstruction[pt], poly_type[pt], m_sample.z, \
                                       m_sample.w, eQT, quartic - cubic_poly.value[:,0] )


                        basis            = err_quartic.getBasis(5, m_sample.z, gp)
                        errQ_exp_by_mode = numpy.zeros((5, gp))
                        for k in range(gp):
                            for j in range(5):
                                errQ_exp_by_mode[j, k] = err_quartic.node[j] * basis[k,j]
                        plt.subplot(1,3,1)
                        errAn.myplot(m_sample.z, quartic, xp, 0, 'x^4')
                        errAn.myplot(m_sample.z, cubic_poly.value[:,0],     xp, 1, 'P(x^4)_3')
                        plt.subplot(1,3,2 )
                        errAn.myplot(m_sample.z, err_quartic.value[:,0],     xp, 0, 'x^4 - P(x^4)_3')
                        for j in range(5):
                            errAn.myplot(m_sample.z, errQ_exp_by_mode[j,:],     xp, j, 'mode '+str(j))
                        plt.legend()
                        plt.subplot(1,3,3 )
                        for j in range(4):
                            errAn.myplot(m_sample.z, errQ_exp_by_mode[j,:],     xp, j, 'mode '+str(j))
                        plt.legend()
                        plt.show()


                    frenet_frame = True
                    errorWeight  = False
                    for j in range(gp):
                        e_dim[j] = m_sample.aot[e,j] - m_sample.x[e,j]
                        e_sig[j] = m_sample.err[e,j]
                        if (errorWeight == True):
                            dotdx[j] = numpy.sqrt(numpy.dot(m_sample.dx[e,j,:,0], m_sample.dx[e,j,:,0]))
                            e_dim[j] *= dotdx[j]
                            e_sig[j] *= dotdx[j]

                    if (frenet_frame == True):

                        T  = numpy.zeros((gp,2))
                        N  = numpy.zeros((gp,2))

                        an = numpy.zeros(gp)
                        at = numpy.zeros(gp)
                        xn = numpy.zeros(gp)
                        xt = numpy.zeros(gp)
                        en = numpy.zeros(gp)
                        et = numpy.zeros(gp)

                        # Returns coordinates of x in the basis TN
                        def frenet_base(T, N, x):
                            det = T[0] * N[1] - T[1] * N[0]
                            IM  = [ [ N[1], -N[0]],
                                    [-T[1],  T[0]] ]
                            a   = IM[0][0] * x[0] + IM[0][1] * x[1]
                            b   = IM[1][0] * x[0] + IM[1][1] * x[1]
                            return a / det, b / det

                        # Returns normal N from T
                        def frenet_n(T):
                            return [-T[1], T[0] ]

                        va  = vf.xnorm(dim, gp, m_sample.daot[e,:,:,0])
                        vx  = vf.xnorm(dim, gp, m_sample.dx[e,:,:,0])

                        for j in range (gp):

                            #T[j,:] = m_sample.daot[e,j,:,0] / va[j] # base alpha aot = alpha o t , daot = alpha ' * dt
                            T[j,:] = m_sample.dx[e,j,:,0] / vx[j]
                            N[j]   = frenet_n(T[j])
                            # Get coordinates alpha
                            at[j], an[j] = frenet_base(T[j], N[j], m_sample.aot[e,j])

                            # Get coordinates e = x - alpha
                            et[j], en[j] = frenet_base(T[j], N[j], m_sample.x[e,j] - m_sample.aot[e,j])

                            vala = at[j] * T[j] + an[j] * N[j]
                            vale = et[j] * T[j] + en[j] * N[j]

                            da = vf.xnorm(2,1,vala.T - m_sample.aot[e,j])
                            de = vf.xnorm(2,1,vale.T - (m_sample.x[e,j] - m_sample.aot[e,j]))
                            if (da > 1.e-12 or de > 1.e-12):
                                print(' caghada! Failed to change basis !!?? ',d1, d2)
                                quit(1)
                            if (numpy.abs(et[j]) > 1.e-12 ):
                                print(' Tangent error not zero!!  et = ', et[j])

                            e_dim[j,0] = et[j]
                            e_dim[j,1] = en[j]
                            e_sig[j]   = en[j]#vf.signed_norm(2, 1, e_dim[j], m_sample.dx[e,j])

                    errAn.myplot(m_sample.z,  et, xp, 0, 'eT')
                    errAn.myplot(m_sample.z,  en, xp, 1, 'eN')
                    plt.legend()
                    plt.show()





                    polyE_dim = polynomial.polynomial(dim, pE, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, e_dim )
                    polyE_sig = polynomial.polynomial(1,   pE, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, e_sig)
                    polyE_high = polynomial.polynomial(1,   pS, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, e_sig)
                    polyE_low = polynomial.polynomial(1,   pE- 1, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, e_sig)
                    # Now compute signed error from 2D error
                    poly_error_sigdim = vf.signed_norm(dim, gp, polyE_dim.value, m_sample.dx[e])

                    showErrorDifferences = False
                    if (showErrorDifferences == True):
                        # Now compute signed error from 2D error
                        plt.subplot(2,1,1)
                        errAn.myplot(m_sample.z,  m_sample.err[e], xp, 0, 'delta')
                        errAn.myplot(m_sample.z, polyE_sig.value[:,0],     xp, 1, 'delta_2p ')
                        errAn.myplot(m_sample.z, polyE_high.value[:,0],     xp, 2, 'delta_np ')
                        errAn.myplot(m_sample.z, polyE_low.value[:,0],     xp, 3, 'delta_2p-1 ')
                        plt.legend()

                        plt.legend()
                        plt.subplot(2,1,2)
                        errAn.myplot(m_sample.z,  m_sample.err[e] -  polyE_sig.value[:,0], xp, 0, 'err delta' + str(pE))
                        errAn.myplot(m_sample.z,  m_sample.err[e] - polyE_high.value[:,0], xp, 1, 'err delta_' + str(pS))
                        errAn.myplot(m_sample.z,  m_sample.err[e] - polyE_low.value[:,0],  xp, 2, 'err delta_' + str(pE - 1))
                        plt.legend()
                        plt.show()
                    sumXA   = 0.0
                    for j in range(gp):
                        zex       [   gp * e + j] = 0.5 * ( (eBD[e + 1] - eBD[e]) * m_sample.z[j] + eBD[e + 1] + eBD[e] )
                        x         [   gp * e + j] = m_sample.x[e,j]
                        t         [   gp * e + j] = m_sample.t[e,j]
                        aot       [   gp * e + j] = m_sample.aot[e,j]
                        errC      [   gp * e + j] = e_dim[j]
                        errT      [   gp * e + j] = e_sig[j]
                        x_poly    [pt,gp * e + j] = polyX.value[j]
                        aot_poly  [pt,gp * e + j] = polyA.value[j]
                        errC_poly [pt,gp * e + j] = polyE_dim.value[j]
                        errT_poly [pt,gp * e + j] = polyE_sig.value[j]
                        errTT_poly[pt,gp * e + j] = poly_error_sigdim[j]
                        derx    = vf.xnorm(dim, 1, m_sample.dx[e,j])
                        err_int = abs(m_sample.err[e,j])
                        wf = 1.0
                        if (eQT == quadratures.eGC): wf = numpy.sqrt(1.0 - m_sample.z[j] * m_sample.z[j])
                        sumXA  += err_int * err_int * derx * m_sample.w[j] * wf
                    disparity_repro[pt,ref] += sumXA * 0.5

                    if e == 0: plt_tits.append(polyX.getType())

                    dist  = vf.mydistance(dim, gp,  m_sample.x[e], polyX.value)
                    if (dist.sum(axis=0)> 1.e-12):
                        print(' !!!! Error in ', polyX.getType(), ' representation = ', ds1)

                    # PLOT MODES IN LOGSCALE FOR ERROR, MESH AND CURVE

                    showModes = True
                    if showModes == True and ref == 0:
                        name     = None
                        outModes = True
                        poly = polyE_sig
                        name = 'SignedError'
                        errAn.plot_modes(xp, f_mode + pt, poly, pX, 0, e, name,  visType )
                        poly = polyE_dim
                        name = 'dimError'
                        errAn.plot_modes(xp, f_mode + pt, poly, pX, 1, e, name,  visType )

                    basisE    = polyE_sig.getBasis(pE + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1
                    basisEdim = polyE_dim.getBasis(pE + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1
                    basisA    = polyA.getBasis    (pS + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1
                    basisX    = polyX.getBasis    (pX + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1

                    # break expansion
                    showBasis = True
                    if showBasis == True and e == 0 and ref == 0:
                        errAn.plot_basis(xp, f_poly,  n_poly_tests, pt + 1, polyE_sig.getType(), m_sample.z, basisE, pE + 1)
                    for k in range(gp):
                        for j in range(pS + 1):
                            aot_exp_by_mode[pt, j, gp * e  + k] = polyA.node[j] * basisA[k,j]
                            if j <= pE:
                                e_exp_by_mode    [pt, j, gp * e  + k]  = polyE_sig.node[j] * basisE[k,j]
                                e_exp_by_mode_dim[pt, j, gp * e  + k]  = polyE_dim.node[j] * basisEdim[k,j]
                            if j <= pX:
                                x_poly_exp_by_mode[pt, j, gp * e  + k] = polyX.node[j] * basisX[k,j]

            if ref != 0: continue

            zAXIS  = numpy.zeros(n + 1)
            yAXIS  = numpy.zeros(n + 1)
            xEP    = numpy.zeros([n + 1,dim])
            for i in range(n):
                zAXIS[i] = zex  [i * gp][0]
                xEP  [i] = x    [i * gp]
            zAXIS[n] = zex  [-1][0]
            xEP[n]   = x    [-1]
            print(' start fig count at ', fcount)
            if (visType == 1):
                fcount = errAn.compare_l2_projection(fcount, polyA.getType(), xp, dim, zex, x, x_poly[pt])
                fcount = errAn.plot_curve(fcount, n, pltInfo, xp, dim, zex, aot, 'alpha o t', x, ' x')

            # Error plots
            for pt in (range(n_poly_tests)):
                fcount = errAn.error_plots(1, fcount, 'error', visType, showLeg, xp, n_poly_tests, pt, pE, pX, zex, \
                                    zAXIS, yAXIS, errT, errT_poly[pt], e_exp_by_mode[pt])
                if dim == 2:
                    fcount = errAn.error_plots(dim, fcount, 'error', visType, showLeg, xp, n_poly_tests, pt, pE, pX, zex, \
                                    zAXIS, yAXIS, errC, errC_poly[pt], e_exp_by_mode_dim[pt])
                    #fcount = errAn.error_plots(dim, fcount, 'alpha', visType, showLeg, xp, n_poly_tests, pt, pE, pX, zex, \
                    #                zAXIS, yAXIS, aot, aot_poly[pt], aot_exp_by_mode[pt])
                fcount = errAn.error_decomp(dim, fcount, xp, plt_tits[pt], visType, pE, zex,
                                    errT_poly[pt], e_exp_by_mode[pt], errC_poly[pt], e_exp_by_mode_dim[pt], zAXIS, yAXIS)
                fcount = errAn.error_decomp_bis(dim, fcount, xp, plt_tits[pt], visType, pE, zex,
                                    errT_poly[pt], e_exp_by_mode[pt], errC_poly[pt], e_exp_by_mode_dim[pt], zAXIS, yAXIS)
        errAn.convergence_IO(nR, ne, disparity, pX, pT, ' DISPARITY ORIGINAL MESH')
        for pt in range(n_poly_tests):
            name='Disparity using a ' + reconstruction[pt] + ' basis and ' + poly_type[pt] + \
                ' polynomials. Solving with ' + quadrature[pt]+ ' rules'
            errAn.convergence_IO(nR, ne, disparity_repro[pt], pX, pT, name)
        if (visType == 1): plt.show()





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
