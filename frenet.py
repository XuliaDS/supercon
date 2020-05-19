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

eScreen = 1
eEPS    = 0



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
            visType = eEPS
        elif showPlots == 1:
            showLeg = True
            visType = eScreen
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
            errTN = numpy.empty((n * gp,dim)) # error components
            errD  = numpy.empty((n * gp    )) # error components

            x_poly     = numpy.empty((n_poly_tests,n * gp,dim))
            aot_poly   = numpy.empty((n_poly_tests,n * gp,dim))

            errTN_poly = numpy.empty((n_poly_tests,n * gp,dim))
            errD_poly  = numpy.empty((n_poly_tests,n * gp    ))

            et_exp_by_mode     = numpy.zeros((n_poly_tests, pE + 1, n * gp))
            en_exp_by_mode     = numpy.zeros((n_poly_tests, pE + 1, n * gp))
            ed_exp_by_mode     = numpy.zeros((n_poly_tests, pE + 1, n * gp))

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
                    # try l2 large t
                    polyA     = polynomial.polynomial(dim, pT, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, m_sample.aot[e])


                    diff_x  = abs(polyX.value   - m_sample.x[e])
                    diff_dx = abs(polyX.tangent - m_sample.dx[e,:,:,0]  )
                    print(' error in x ', diff_x.sum(axis=0), ' error in dx ', diff_dx.sum(axis=0))
                    diff_t  = abs(polyA.value   - m_sample.aot[e])
                    diff_dt = abs(polyA.tangent - m_sample.daot[e,:,0]  )
                    print(' error in x ', diff_x.sum(axis=0), ' error in dx ', diff_dx.sum(axis=0))

                    frenet_frame = True
                    if (frenet_frame == True):

                        Ta    = numpy.zeros((gp,2))
                        Na    = numpy.zeros((gp,2))
                        Tx    = numpy.zeros((gp,2))
                        Nx    = numpy.zeros((gp,2))

                        a_fa  = numpy.zeros((gp, 2))
                        x_fa  = numpy.zeros((gp, 2))
                        e_fa  = numpy.zeros((gp, 2))
                        da_fa = numpy.zeros((gp, 2))
                        dx_fa = numpy.zeros((gp, 2))
                        de_fa = numpy.zeros((gp, 2))

                        a_fx  = numpy.zeros((gp, 2))
                        x_fx  = numpy.zeros((gp, 2))
                        e_fx  = numpy.zeros((gp, 2))
                        da_fx = numpy.zeros((gp, 2))
                        dx_fx = numpy.zeros((gp, 2))
                        de_fx = numpy.zeros((gp, 2))

                        tx_dot_ta = numpy.zeros(gp)  # angle between rereference systems
                        tx_dot_na = numpy.zeros(gp)  # angle between rereference systems

                        # Returns coordinates of x in the basis TN
                        def frenet_base_2d(T, N, x):
                            return [ numpy.dot(T,x), numpy.dot(N,x)]

                        # Returns normal N from T
                        def frenet_n_2d(T):
                            return [-T[1], T[0] ]

                        Jx      = vf.xnorm(dim, gp, m_sample.dx[e])
                        Ja      = vf.xnorm(dim, gp, m_sample.daot[e])
                        kappa_x = numpy.zeros(gp)
                        kappa_a = numpy.zeros(gp)
                        txta    = numpy.zeros(gp)
                        txna    = numpy.zeros(gp)
                        for j in range (gp):
                            Ta[j] = m_sample.daot[e,j,:,0] / Ja[j] # base alpha aot = alpha o t , daot = alpha ' * dt
                            Tx[j] =   m_sample.dx[e,j,:,0] / Jx[j]
                            Na[j] = frenet_n_2d(Ta[j])
                            Nx[j] = frenet_n_2d(Tx[j])

                            txta[j] = numpy.dot(Tx[j], Ta[j])
                            txna[j] = numpy.dot(Tx[j], Na[j])

                            kappa_x[j] = abs(polyX.tangent[j,0] * polyX.hessian[j,1] - polyX.tangent[j,1] * polyX.hessian[j,0]) / numpy.power(Jx[j],3)
                            kappa_a[j] = abs(polyA.tangent[j,0] * polyA.hessian[j,1] - polyA.tangent[j,1] * polyA.hessian[j,0]) / numpy.power(Ja[j],3)


                            # Get coordinates base alpha
                            a_fa[j]  = frenet_base_2d(Ta[j], Na[j], m_sample.aot[e,j])
                            x_fa[j]  = frenet_base_2d(Ta[j], Na[j],   m_sample.x[e,j])
                            e_fa[j]  = x_fa[j] - a_fa[j]

                            da_fa[j] = [ Ja[j] + kappa_a[j] * a_fa[j,1],
                                               - kappa_a[j] * a_fa[j,0] ]
                            dx_fa[j] = [ Jx[j] * txta[j] + kappa_a[j] * x_fa[j,1],
                                         Jx[j] * txna[j] - kappa_a[j] * x_fa[j,0]]
                            de_fa[j] = dx_fa[j] - da_fa[j]

                            # Get coordinates base x
                            a_fx[j]  = frenet_base_2d(Tx[j], Nx[j], m_sample.aot[e,j])
                            x_fx[j]  = frenet_base_2d(Tx[j], Nx[j], m_sample.x[e,j])
                            e_fx[j]  = x_fx[j] - a_fx[j]

                            da_fx[j] = [ Ja[j] * txna[j] + kappa_x[j] * a_fx[j,1],
                                         Ja[j] * txna[j] - kappa_x[j] * a_fx[j,0]]
                            dx_fx[j] = [ Jx[j] + kappa_x[j] * x_fx[j,1],
                                               - kappa_x[j] * x_fx[j,0] ]

                            de_fx[j] = dx_fx[j] - da_fx[j]

                            c1 = dx_fa[j,0] * Ta[j] + dx_fa[j,1] * Na[j]
                            c2 = da_fx[j,0] * Tx[j] + da_fx[j,1] * Nx[j]

                            da = vf.xnorm(2,1,c1.T -   m_sample.dx[e,j,:,0])
                            de = vf.xnorm(2,1,c2.T - m_sample.daot[e,j,:,0])
                            if (da > 1.e-12 or de > 1.e-12):
                                print(' caghada! Failed to change basis !!?? ',da, de)

                        plt.show()
                        P_0p    = polyX.getBasis(pX + 1, m_sample.z, gp)
                        # check 1
                        lhs  = numpy.zeros((gp,2))
                        rhs  = numpy.zeros((gp,2))
                        sumX = 0.0
                        for j in range (gp):

                            #lhs[j] =  dx_fa[j,0] * Ta[j] + dx_fa[j,1] * Na[j]
                            #rhs[j] = m_sample.dx[e,j,:,0]
                            lhs[j] = (x_fa[j,1] - a_fa[j,1]) * kappa_x[j]#* e_fa[j,0 ]#de_fa[j,1]

                            rhs[j] =  - 2.0 * txna[j]  # * e_fa[j,0]#Jx[j] * txna[j] - kappa_a[j] * e_fa[j,0]#Jx[j] * txna[j] #e_fa[j,1] * kappa_x[j]
                            #rhs[j] = kappa_a[j] * (-e_fa[j,0])#     -2.0 * txta[j]
                            print( ' lhs ', lhs[j], ' rhs ', rhs[j])
                            sumX += (lhs[j] - rhs[j] )* m_sample.w[j]
                        print(' sum is ', sumX)
                        plt.subplot(2,1,1)
                    #    errAn.myplot(m_sample.z, lhs[:,0], xp,0, 'TxTa')
                        errAn.myplot(m_sample.z, lhs[:,0], xp,0, 'TxTa')
                        errAn.myplot(m_sample.z, rhs[:,0], xp,1, 'TxTa')
                        #errAn.myplot(m_sample.z, rhs[:,0], xp,1, 'TxTa')
                        plt.subplot(2,1,2)
                        #errAn.myplot(m_sample.z, lhs[:,1], xp,0, 'TxTa')
                        errAn.myplot(m_sample.z, rhs[:,0] - lhs[:,0], xp,1, 'TxTa')

                        plt.show()

                        plt.subplot(2,1,1)
                        errAn.myplot(m_sample.z, lhs[:,0], xp,0, 'TxTa')
                        errAn.myplot(m_sample.z, rhs[:,0], xp,1, 'TxTa')
                        errAn.myplot(m_sample.z, -Na[:,0], xp,3, 'TxTa')

                        plt.subplot(2,1,2)
                        errAn.myplot(m_sample.z, lhs[:,1], xp,0, 'TxTa')
                        errAn.myplot(m_sample.z, rhs[:,1], xp,1, 'TxTa')
                        errAn.myplot(m_sample.z, -Na[:,1], xp,3, 'TxTa')

                        plt.show()
                        basisQ = polyX.getBasis(pT + 1, m_sample.z, gp)
                        for j in range (gp):
                            lhs[j] = txta[j] * txna[j]
                            rhs[j] = extn[j,0] * 0.5 * kappa[j]
                        for i in range(pT + 1):
                            sumL   = 0.0
                            sumR   = 0.0
                            for j in range (gp):
                                sumL += lhs[j] * basisQ[j][i] * m_sample.w[j]
                                sumR += rhs[j] * basisQ[j][i] * m_sample.w[j]
                            print(' i  = ',' integrals ', sumL, sumR)

                        for j in range (gp):

                            lhs[j] = txna[j] / txta[j] * Jx[j] * kappa[j] * (eatn[j,0]**2 + eatn[j,1]**2)
                            rhs[j] = -deatn[j,0] * eatn[j,0] - deatn[j,1] * eatn[j,1]
                            lhs[j] =  Jx[j] * txna[j]
                            rhs[j] = deatn[j,1]

                        plt.subplot(1,2,1)
                        errAn.myplot(m_sample.z, lhs, xp,0, 'TxTa')
                        errAn.myplot(m_sample.z, lhs, xp,1, 'TxTa')
                        plt.subplot(1,2,2)
                        errAn.myplot(m_sample.z, lhs - rhs, xp,0, 'TxTa')

                        plt.show()
                        quit()

                        rs_e    = numpy.zeros(gp)
                        rs_de   = numpy.zeros(gp)
                        ek      = numpy.zeros(gp)
                        l2d     = numpy.zeros((gp, dim))
                        r2d     = numpy.zeros((gp, dim))
                        eq_t    = numpy.zeros(gp)
                        eq_n    = numpy.zeros(gp)

                        for j in range (gp):
                            #kappa[j] = 1.0 / abs(polyX.tangent[j,0] * polyX.hessian[j,1] - polyX.tangent[j,1] * polyX.hessian[j,0]  )
                            kappa[j] = abs(polyX.tangent[j,0] * polyX.hessian[j,1] - polyX.tangent[j,1] * polyX.hessian[j,0]) / numpy.power(Jx[j],3)
                            dx       = numpy.sqrt(polyX.tangent[j,0]**2 + polyX.tangent[j,1] **2 )
                            ek[j]    = 0.5 * etn[j,1] * kappa[j] * Jx[j]

                            rs_e[j]  = detn[j,1] * txta[j] + ek[j] * txna[j]
                            rs_de[j] = detn[j,1] * txna[j] - ek[j] * txta[j]

                            t1       = (etn[j,0] * T[j] + etn[j,1] * N[j]) * Jx[j]
                            t2       = (detn[j,0] * etn[j,0] + detn[j,1] * etn[j,1] ) * Tx[j]
                            t3       = 0.5 * (etn[j,0]** 2 + etn[j,1]** 2) * Jx[j] * kappa[j] * Nx[j]
                            l2d[j]   = t1 - t2 + t3##detn[j,1] * Tx[j]   - ek[j] * Nx[j]
                            r2d[j,0] = numpy.dot(l2d[j], T[j])
                            r2d[j,1] = numpy.dot(l2d[j], N[j])
                            eq_t[j]  = txta[j] * txna[j]
                            eq_n[j]  = 0.5 * etn[j,0] * kappa[j]


                        plt.subplot(1,2,1)
                        errAn.myplot(m_sample.z, eq_t, xp,0, 'TxTa')
                        plt.subplot(1,2,2)
                        errAn.myplot(m_sample.z, eq_n, xp,1, 'TxTa')
                        plt.show()





                        plt.subplot(1,2,1)
                        errAn.myplot(m_sample.z, rs_e, xp,0, 'TxTa')
                        errAn.myplot(m_sample.z,  detn[:,1] * txta, xp,1, 'TxTa')

                        plt.subplot(1,2,2)
                        errAn.myplot(m_sample.z, rs_e - detn[:,1], xp,0, 'TxTa')

                        plt.legend()
                        plt.show()

                        plt.subplot(1,2,1)
                        errAn.myplot(m_sample.z, abs(rs_de), xp,0, 'TxTa')
                        errAn.myplot(m_sample.z, abs(ek * txta) , xp,1, 'TxTa')
                        #errAn.myplot(m_sample.z, Jx, xp,2, 'TxTa')


                        plt.subplot(1,2,2)
                        errAn.myplot(m_sample.z, rs_e - ek * txta, xp,0, 'TxTa')
                        #errAn.myplot(m_sample.z, rs_de,xp, 1, 'TxNa')
                        plt.legend()
                        plt.show()
                        for i in range (pX + 1):
                            l2L_de = 0
                            l2R_de = 0
                            l2L_e  = 0
                            l2R_e  = 0
                            for j in range (gp):
                                l2L_de += detn[j,1] * P_0p[j,i] * m_sample.w[j]
                                l2L_e  +=  etn[j,1] * P_0p[j,i] * m_sample.w[j]

                                l2R_de += rs_de[j] * P_0p[j,i] * m_sample.w[j]
                                l2R_e  +=  rs_e[j] * P_0p[j,i] * m_sample.w[j]

                            print ('  equation en   mode = ', i, '--> ',l2L_e,'=', l2R_e,' error ', numpy.abs(l2L_e - l2R_e))
                            print ('  equation edot mode = ', i, '--> ',l2L_de,'=', l2R_de,' error ', numpy.abs(l2L_de - l2R_de))

                        if (visType == eScreen) :
                            plt.subplot(2,2,1)
                            errAn.myplot(m_sample.z, detn[:,1],xp, 0, 'edot_n')
                            errAn.myplot(m_sample.z, rs_de,xp, 2, 'Jx * sin(theta)')
                            errAn.fig_params('%1.1f', '%1.e',m_sample.z,  detn[:,1], m_sample.z, rs_de , 2, '$\\xi$')
                            plt.xticks([], [])
                            plt.xlabel(None)
                            plt.legend()
                            plt.subplot(2,2,2)
                            errAn.myplot(m_sample.z, detn[:,1] - rs_de,xp, 2, 'error')
                            errAn.fig_params1('%1.1f', '%1.e',m_sample.z, abs( detn[:,1] - rs_de)  , 2, '$\\xi$')
                            plt.xticks([], [])
                            plt.xlabel(None)
                            plt.legend()
                            plt.subplot(2,2,3)
                            errAn.myplot(m_sample.z, etn[:,1],xp, 0, 'e_n')
                            errAn.myplot(m_sample.z, rs_e,xp, 2, 'kappa * Jx')
                            errAn.fig_params('%1.1f', '%1.e',m_sample.z,  etn[:,1], m_sample.z, rs_e , 2, '$\\xi$')
                            plt.xticks([], [])
                            plt.xlabel(None)
                            plt.legend()
                            plt.subplot(2,2,4)
                            errAn.myplot(m_sample.z, etn[:,1] - rs_e,xp, 2, 'error')
                            errAn.fig_params1('%1.1f', '%1.e',m_sample.z, abs(etn[:,1] - rs_e) , 2, '$\\xi$')
                            plt.xticks([], [])
                            plt.xlabel(None)
                            plt.legend()
                            plt.show()
                        else:
                            errAn.fig_header(fcount, visType)
                            errAn.myplot(m_sample.z, detn[:,1],xp, 0, 'edot_n')
                            errAn.myplot(m_sample.z, rs_de,xp, 2, 'Jx * sin(theta)')
                            errAn.fig_params('%1.1f', '%1.e',m_sample.z,  detn[:,1], m_sample.z, rs_de , 2, '$\\xi$')
                            plt.xticks([], [])
                            plt.xlabel(None)
                            plt.savefig('results/edot_weak_p' + str(pX) +'.eps', bbox_inches='tight', pad_inches=0)
                            plt.close()
                            errAn.myplot(m_sample.z, detn[:,1],xp, 0, 'edot_n')
                            errAn.myplot(m_sample.z, rs_de,xp, 2, 'Jx * sin(theta)')
                            errAn.fig_params('%1.1f', '%1.e',m_sample.z, abs( detn[:,1]), m_sample.z, rs_de , 2, '$\\xi$')
                            plt.xticks([], [])
                            plt.xlabel(None)
                            plt.savefig('results/abs_edot_weak_p' + str(pX)+'.eps', bbox_inches='tight', pad_inches=0)
                            plt.close()
                            errAn.myplot(m_sample.z, detn[:,1] - rs_de,xp, 2, 'error')
                            errAn.fig_params1('%1.1f', '%1.e',m_sample.z, abs( detn[:,1])- rs_de , 2, '$\\xi$')
                            plt.xticks([], [])
                            plt.xlabel(None)
                            plt.savefig('results/edot_err_p' + str(pX)+'.eps', bbox_inches='tight', pad_inches=0)
                            plt.close()

                            quit()

                    polyET = polynomial.polynomial(1, pE, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, etn[:,0] )

                    polyEN = polynomial.polynomial(1, pE, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, etn[:,1] )

                    polyED = polynomial.polynomial(1, pE, reconstruction[pt], poly_type[pt], m_sample.z, \
                                                       m_sample.w, eQT, m_sample.err[e])

                    sumXA  = 0.0
                    for j in range(gp):

                        zex        [   gp * e + j]   = 0.5 * ( (eBD[e + 1] - eBD[e]) * m_sample.z[j] + eBD[e + 1] + eBD[e] )
                        x          [   gp * e + j]   = m_sample.x[e,j]
                        x_poly     [pt,gp * e + j]   = polyX.value[j]

                        t          [   gp * e + j]   = m_sample.t[e,j]

                        aot        [   gp * e + j]   = m_sample.aot[e,j]
                        aot_poly   [pt,gp * e + j]   = polyA.value[j]

                        errTN      [   gp * e + j]   = etn[j]
                        errTN_poly [pt,gp * e + j,0] = polyET.value[j]
                        errTN_poly [pt,gp * e + j,1] = polyEN.value[j]

                        errD       [   gp * e + j]   = m_sample.err[e,j]
                        errD_poly  [pt,gp * e + j]   = polyED.value[j]

                        derx    = vf.xnorm(dim, 1, m_sample.dx[e,j])
                        err_int = abs(m_sample.err[e,j])
                        sumXA  += err_int * err_int * derx * m_sample.w[j]

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
                        poly     = polyET
                        name     = 'Tangent_Error'
                        errAn.plot_modes(xp, f_mode + pt, 3, poly, pX, 0, e, name,  visType )
                        poly     = polyEN
                        name     = 'Normal_Error'
                        errAn.plot_modes(xp, f_mode + pt, 3, poly, pX, 1, e, name,  visType )
                        poly     = polyED
                        name     = 'Absolute_Error'
                        errAn.plot_modes(xp, f_mode + pt, 3, poly, pX, 2, e, name,  visType )

                    basisET = polyET.getBasis(pE + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1
                    basisEN = polyEN.getBasis(pE + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1
                    basisED = polyED.getBasis(pE + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1

                    basisA  = polyA.getBasis (pS + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1
                    basisX  = polyX.getBasis (pX + 1, m_sample.z, gp) # degree 2 * p => order 2 * p +1

                    # break expansion
                    showBasis = True
                    if showBasis == True and e == 0 and ref == 0 and visType == eScreen :
                        errAn.plot_basis(xp, f_poly,  n_poly_tests, pt + 1, polyET.getType(), m_sample.z, basisET, pE + 1)

                    for k in range(gp):
                        for j in range(pS + 1):
                            aot_exp_by_mode[pt, j, gp * e  + k] = polyA.node[j] * basisA[k,j]
                            if j <= pE:
                                et_exp_by_mode[pt, j, gp * e  + k]  = polyET.node[j] * basisET[k,j]
                                en_exp_by_mode[pt, j, gp * e  + k]  = polyEN.node[j] * basisEN[k,j]
                                ed_exp_by_mode[pt, j, gp * e  + k]  = polyED.node[j] * basisED[k,j]

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
            if (visType == eScreen):
                fcount = errAn.compare_l2_projection(fcount, polyA.getType(), xp, dim, zex, x, x_poly[pt])
                fcount = errAn.plot_curve(fcount, n, pltInfo, xp, dim, zex, aot, 'alpha o t', x, ' x')

            # Error plots
            for pt in (range(n_poly_tests)):

                fcount = errAn.compare_errors_tnb(fcount, visType, xp, zex, errTN[:,0], errTN[:,1], errD)

                fcount = errAn.error_plots(1, fcount, 'Tangent_error', visType, showLeg, xp, n_poly_tests, pt, pE, pX, zex, \
                                    zAXIS, yAXIS, errTN[:,0], errTN_poly[pt,:,0], et_exp_by_mode[pt])

                fcount = errAn.error_plots(1, fcount, 'Normal_error', visType, showLeg, xp, n_poly_tests, pt, pE, pX, zex, \
                                    zAXIS, yAXIS, errTN[:,1], errTN_poly[pt,:,1], en_exp_by_mode[pt])

                fcount = errAn.error_plots(1, fcount, 'Absolute_error', visType, showLeg, xp, n_poly_tests, pt, pE, pX, zex, \
                                    zAXIS, yAXIS, errD, errD_poly[pt], ed_exp_by_mode[pt])

                fcount = errAn.error_decomp(1, fcount, xp, plt_tits[pt], visType, pE, zex,errTN_poly[pt,:,0],\
                                et_exp_by_mode[pt], errTN_poly[pt,:,0], et_exp_by_mode[pt], zAXIS, yAXIS, 'Tangent_Error')

                fcount = errAn.error_decomp(1, fcount, xp, plt_tits[pt], visType, pE, zex, errTN_poly[pt,:,1], \
                                en_exp_by_mode[pt], errTN_poly[pt,:,1], en_exp_by_mode[pt], zAXIS, yAXIS, 'Normal_Error')

                fcount = errAn.error_decomp(1, fcount, xp, plt_tits[pt], visType, pE, zex,errD_poly[pt], \
                                ed_exp_by_mode[pt], errD_poly[pt], ed_exp_by_mode[pt], zAXIS, yAXIS, 'Absolute_Error')


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
