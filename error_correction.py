#!/bin/python -u
import numpy
import math
import unittest
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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



def orientation(x, y):
    det = x[0] * y[1] - y[0] * x[1]      # determinant
    if (det > 0): return 1.0
    return -1.0

def my2dnorm (sizex, x, dx):
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
    if dim == 1:    return abs(x - y)
    else:
        z = numpy.zeros(n)
        for i in range(n):
            for d in range(dim):
                z[i] += (x[i,d] - y[i,d]) ** 2
            z[i] = numpy.sqrt(z[i])
        return z


def convergence_IO(nR, ne, value, pX, pT, title):
    print("____________________________________________________________________\n")
    print("----------------------- POLYNOMIAL DEGREES: X ",pX," T ",pT," ----------------")
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

    @staticmethod
    def testDistanceFunction(dim, pX, pT, ne, nR, curve, I, showPlots):

        relocateX = False
        fixU      = False
        callFix   = True
        method    = 'Newton'
        tolDistanceCalculation = 1.e-8
        showMesh0 = 1
        tol = 1.e-8

        disparity_e           = numpy.zeros([2,nR + 1])
        disparity_XA          = numpy.zeros([2,nR + 1])
        disparity_DXAT        = numpy.zeros([2,nR + 1])
        gp                    = 80
        l2gp                  = 24
        objectiveFunctionName = "Intrinsic"
        frechetFunctionName   = "Intrinsic"
        if (dim == 1): parametrization = TestDistanceFunctionOptimization.getGeometry1D(curve, I[0], I[1])
        else:          parametrization = TestDistanceFunctionOptimization.getGeometry2D(curve, I[0], I[1])

        figcount = 1
        ea       = numpy.zeros(4)
        dea      = numpy.zeros(2)
        pltInfo  = '    pX = '+str(pX)+' pT = '+str(pT)

        gpx, wp = quadratures.qType(pX + 1, quadratures.eLGL)
        gpt, wt = quadratures.qType(pT + 1, quadratures.eLGL)

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
                fixU      = fixU
                )

            meshO, meshI   = optimizer.run()
            n              = meshI.theNOfElements

            newMasterElementX       = meshO.theMasterElementMakerX.createMasterElement(pX, l2gp - 1)
            newMasterElementU       = meshO.theMasterElementMakerU.createMasterElement(pT, l2gp - 1)

            meshO.theMasterElementX = newMasterElementX
            meshO.theMasterElementU = newMasterElementU

            l2w   = meshO.theMasterElementX.theGaussWeights
            l2z   = meshO.theMasterElementX.theGaussPoints

            x_forL2     = numpy.empty((2, n, l2gp,dim))
            dx_forL2    = numpy.empty((2, n, l2gp,dim, 1))

            alpha_forL2 = numpy.empty((2, n, l2gp,dim))

            for i in range (n):
                x_forL2[0,i]     = meshO.getXElement(i)
                dx_forL2[0,i]    = meshO.getDNXElement(i)
                tt               = meshO.getUElement(i)
                alpha_forL2[0,i] = parametrization.value(tt)


            newMasterElementX       = meshO.theMasterElementMakerX.createMasterElement(pX, gp - 1)
            newMasterElementU       = meshO.theMasterElementMakerU.createMasterElement(pT, gp - 1)
            oldMasterElementX       = meshI.theMasterElementX

            meshO.theMasterElementX = newMasterElementX
            meshO.theMasterElementU = newMasterElementU

            meshI.theMasterElementX = newMasterElementX
            meshI.theMasterElementU = newMasterElementU

            w   = meshI.theMasterElementX.theGaussWeights
            z   = meshI.theMasterElementX.theGaussPoints

            zx  = numpy.empty((n *  gp     ,1))
            zp  = numpy.empty((n * (pX + 1),1))
            zu  = numpy.empty((n * (pT + 1),1))

            eBD = numpy.zeros(n + 1)
            for i in range(n + 1):
                 eBD[i] = parametrization.theT0 + h * i

            x     = numpy.empty((2,n * gp,dim))
            t     = numpy.empty((2,n * gp,dim))
            alpha = numpy.empty((2,n * gp,dim))
            ep_2p = numpy.empty((2,n * gp    ))
            errC  = numpy.empty((2,n * gp,dim))
            errT  = numpy.empty((2,n * gp    ))
            polC  = numpy.empty((2,n * gp,dim))
            polT  = numpy.empty((2,n * gp    ))
            aPF   = numpy.empty((  n * gp,dim))
            dumb  = numpy.zeros([1,1])
            for type in range(showMesh0):
                if type == 0: mesh = meshO
                else:         mesh = meshI

                disf,proje,norm    = TestDistanceFunctionOptimization.getMeshDistances(
                                      mesh,parametrization,frechetFunctionName,
                                      tolDistanceCalculation, gp - 1)

                disparity_e[type, ref] = disf * disf * 0.5
                        #create interpolating t
                for i in range(n):
                    if type == 0:
                        for j in range(gp):
                            zx[i * gp + j,0] = 0.5 * ( (eBD[i + 1] - eBD[i]) * z[j] + eBD[i + 1] + eBD[i] )
                            dumb[0]          = zx[i * gp + j]
                            aux              = parametrization.value(dumb)
                            aPF[i * gp + j]  = aux
                            if j < pX + 1:
                                zp[i * (pX + 1) + j,0]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpx[j] + eBD[i + 1] + eBD[i] )
                            if j < pT + 1:
                                zu[i * (pT + 1) + j,0]  = 0.5 * ( (eBD[i + 1] - eBD[i]) * gpt[j] + eBD[i + 1] + eBD[i] )

                    x_i     = mesh.getXElement(i)
                    dx_i    = mesh.getDNXElement(i)
                    t_i     = mesh.getUElement(i)
                    alpha_i = parametrization.value(t_i)

                    idEX       = mesh.theElementsX[i,:]
                    x_nodes    = mesh.theNodes[idEX,:]
                    inter_poly = polynomial.polynomial(dim, pX, polynomial.eMono)
                    inter_poly.interpolate(gpx, x_nodes)
                    aux        = inter_poly.evaluate(z, gp)
                    xint       = aux[:,0]

                    modal_poly = polynomial.polynomial(dim, pX, polynomial.eModal)

                    modal_poly.l2_legPro(x_forL2[0,i], l2z, l2w)
                    aux       = modal_poly.evaluate(z, gp)
                    xmod_l2gp = aux[:,0]
                    xmod_gp   = modal_poly.l2_legPro(x_i, z, w)

                    dm_l2gp = mydistance(dim, gp, x_i, xmod_l2gp)
                    dm_gp   = mydistance(dim, gp, x_i, xmod_gp)
                    di      = mydistance(dim, gp, x_i, xint)



                    dif1  = dm_l2gp.sum(axis=0)
                    dif2  = dm_gp.sum(axis=0)
                    dif3  = di.sum(axis=0)
                    if (dif1 > 1.e-12 or dif2 > 1.e-12 or dif3 > 1.e-12):
                        print(' !!!! L2Projection optimized solution using', l2gp, ' Gauss Points', dif1)
                        print(' !!!! L2Projection optimized solution using',   gp, ' Gauss Points', dif2)
                        print(' !!!! Projection optimized solution onto monomial basis error ', dif3)
                        if showPlots == 2:
                            fig       = plt.figure(figcount)
                            figcount += 1
                            plt.subplot(2,1,1)
                            plt.title(' Error for ' +  modal_poly.getType())
                            plt.plot(z,dm_gp,   c = 'b', linestyle ='-.', label = 'using '+ str(gp) + ' GPs')
                            plt.plot(z,dm_l2gp, c = 'r', linestyle ='-.', label = 'using '+ str(l2gp) + ' GPs')

                            plt.legend()

                            plt.subplot(2,1,2)
                            plt.title(' Error for ' +  inter_poly.getType())
                            plt.plot(z,di,  c = 'b', linestyle ='-.')

                    if dim == 1:
                        if (type == 1): pE = max(pX, pT) + 1
                        else:           pE =          pX + pT
                    else:
                        if (type == 1): pE = pX + 1
                        else:           pE =  2 * pX

                    # Approximate error function by higher order polynomial

                    '''xbar         = numpy.zeros(gp)
                    alphabar_o_t = numpy.zeros(gp)
                    for k in range(gp):
                        xbar[k]         = 0.5 * (    x_i[k,0] +     x_i[k,1])
                        alphabar_o_t[k] = 0.5 * (alpha_i[k,0] + alpha_i[k,1])
                    signed_dist_x_alpha = xbar - alphabar_o_t'''

                #    for k in range(gp):
                #        xbar[k]         = numpy.sqrt( (x_i[k,0] * x_i[k,0] + x_i[k,1] * x_i[k,1]) )
                #        alphabar_o_t[k] = numpy.sqrt( (alpha_i[k,0] * alpha_i[k,0] + alpha_i[k,1] * alpha_i[k,1]) )
                    # CAMBIAR !!!!!!!!

                    poly_dim        = 1
                    super_poly_l2gp = polynomial.polynomial(poly_dim, pE * 3, polynomial.eModal)
                    modal_poly_l2gp = polynomial.polynomial(poly_dim, pE    , polynomial.eModal)

                    super_poly_gp   = polynomial.polynomial(poly_dim, pE * 3, polynomial.eModal)
                    modal_poly_gp   = polynomial.polynomial(poly_dim, pE    , polynomial.eModal)

                    sign_dist_l2gp  = numpy.empty((2, l2gp, poly_dim))
                    sign_dist_gp    = numpy.empty((2,   gp, poly_dim))

                    abs_dist_true = mydistance(dim, gp, x_i, alpha_i)
                    abs_dist_mod  = mydistance(dim, gp, xmod_l2gp, alpha_i)
                    abs_dist_int  = mydistance(dim, gp, xint, alpha_i)

                    if dim == 2:
                        aux                   = my2dnorm(l2gp, x_forL2[0,i] - alpha_forL2[0,i], dx_forL2[0,i])
                        sign_dist_l2gp[0,:,0] = aux

                        aux                   = my2dnorm(gp, x_i - alpha_i, dx_i)
                        sign_dist_gp[0,:,0]   = aux
                    else:
                        sign_dist_l2gp[0] = x_forL2[0,i] - alpha_forL2[0,i]
                        sign_dist_gp[0]   = x_i          - alpha_i

                    for j in range(l2gp):
                        sign_dist_l2gp[1,j] = sign_dist_l2gp[0,j] * l2w[j]
                    for j in range(gp):
                        sign_dist_gp[  1,j] = sign_dist_gp[0,j]   *   w[j]


                    # Create solutions using a modal basis solved with less quadrature points
                    modal_poly_l2gp.l2_legPro(sign_dist_l2gp[0], l2z, l2w)
                    super_poly_l2gp.l2_legPro(sign_dist_l2gp[0], l2z, l2w)

                    aux           = modal_poly_l2gp.evaluate(z, gp)
                    e_mod_2p_l2gp = aux[:,0]

                    aux           = super_poly_l2gp.evaluate(z, gp)
                    e_super_l2gp  = aux[:,0]

                    # Create solutions using a modal basis solved with TOO MANY quadrature points
                    e_mod_2p_gp = modal_poly_gp.l2_legPro(sign_dist_gp[0], z, w)
                    e_super_gp  = super_poly_gp.l2_legPro(sign_dist_gp[0], z, w)

                    # Create a monomial error solution
                    z2p, w2p     = quadratures.qType(pE + 1, quadratures.eLGL)
                    i_nodes_gp   = numpy.zeros([pE + 1, 1])
                    i_nodes_l2gp = numpy.zeros([pE + 1, 1])

                    super_basis_gp   = super_poly_gp.getBasis  (super_poly_gp.n,   z2p, len(z2p)) # degree 2 * p => order 2 * p +1
                    super_basis_l2gp = super_poly_l2gp.getBasis(super_poly_l2gp.n, z2p, len(z2p)) # degree 2 * p => order 2 * p +1

                    for k in range(pE + 1):
                        for j in range(super_poly_gp.n):
                            i_nodes_gp[k]   +=   super_poly_gp.node[j] * super_basis_gp  [k,j]
                            i_nodes_l2gp[k] += super_poly_l2gp.node[j] * super_basis_l2gp[k,j]

                    inter_poly_gp = polynomial.polynomial(1, pE, polynomial.eMono)
                    inter_poly_gp.interpolate(z2p, i_nodes_gp)

                    inter_poly_l2gp = polynomial.polynomial(1, pE, polynomial.eMono)
                    inter_poly_l2gp.interpolate(z2p, i_nodes_l2gp)

                    aux           = inter_poly_gp.evaluate(z, gp)
                    e_int_2p_gp   = aux[:,0]
                    aux           = inter_poly_l2gp.evaluate(z, gp)
                    e_int_2p_l2gp = aux[:,0]

                    pdiff_gp   = e_int_2p_gp   - e_mod_2p_gp
                    pdiff_l2gp = e_int_2p_l2gp - e_mod_2p_l2gp

                    aux1 =   pdiff_gp.sum(axis=0)
                    aux2 = pdiff_l2gp.sum(axis=0)
                    print(' diff  using gp ', aux1, 'diff using l2gp', aux2 )
                    if (aux1 > 1.e-14 or aux2 > 1.e-14):
                        print(' !!! Error changing from ', modal_poly_gp.getType(), ' to ', inter_poly_gp.getType() )
                        print(' diff gp ', aux1, 'diff l2gp ', aux2)
                        if showPlots == 2:

                            fig       = plt.figure(figcount)
                            figcount += 1
                            plt.subplot(2,1,1)
                            plt.plot(z, e_mod_2p_gp  , c = 'r'     , linestyle = '-' , label = modal_poly_gp.getType()   + ' ' +str(gp)   + ' GPs')
                            plt.plot(z, e_int_2p_gp  , c = 'orange', linestyle = '-.' , label = inter_poly_gp.getType()   + ' ' +str(gp)   + ' GPs')
                            plt.plot(z, e_mod_2p_l2gp, c = 'b'     , linestyle = '--', label = modal_poly_l2gp.getType() + ' ' +str(l2gp)   + ' GPs')
                            plt.plot(z, e_int_2p_l2gp, c = 'c'     , linestyle = ':', label = inter_poly_l2gp.getType() + ' ' +str(l2gp)   + ' GPs')

                            plt.autoscale()
                            plt.legend()

                            plt.subplot(2,1,2)
                            plt.plot(z,  pdiff_gp, c = 'r', linestyle ='-.',label = 'error base '+ str(gp  ) + ' GPs')
                            plt.plot(z,pdiff_l2gp, c = 'b', linestyle ='--',label = 'error base '+ str(l2gp) + ' GPs')
                            plt.legend()


                    if showPlots == 2:

                        fig       = plt.figure(figcount)
                        figcount += 1
                        plt.subplot(2,1,1)
                        plt.title(' Error Profile Using ' + str(gp) + ' Quadrature Rule')
                        plt.plot(z, sign_dist_gp[0], linestyle ='-',  label ='true')
                        plt.plot(z, e_mod_2p_gp    , linestyle =':',  label =  modal_poly_gp.getType()  + ' 2p')
                        plt.plot(z, e_int_2p_gp    , linestyle ='--', label =  inter_poly_gp.getType()  + ' 2p')
                        plt.plot(z, e_super_gp     , linestyle ='-.', label =  super_poly_gp.getType()  + ' 6p')
                        plt.scatter(z2p, i_nodes_gp)
                        plt.legend()
                        plt.subplot(2,1,2)
                        plt.title(' Error Profile Using ' + str(l2gp) + ' Quadrature Rule')
                        plt.plot(z, sign_dist_gp[0], linestyle ='-',  label ='true')
                        plt.plot(z, e_mod_2p_l2gp  , linestyle =':',  label =  modal_poly_l2gp.getType()  + ' 2p')
                        plt.plot(z, e_int_2p_l2gp  , linestyle ='--', label =  inter_poly_l2gp.getType()  + ' 2p')
                        plt.plot(z, e_super_l2gp   , linestyle ='-.', label =  super_poly_l2gp.getType()  + ' 6p')
                        plt.scatter(z2p, i_nodes_l2gp)
                        plt.legend()


                    modal_poly     = modal_poly_l2gp
                    inter_poly     = inter_poly_l2gp
                    modal_2p_error = e_mod_2p_l2gp
                    inter_2p_error = e_int_2p_l2gp
                    xmod           = xmod_l2gp
                    true_error     = sign_dist_gp[0]

                    if showPlots == 2:

                        fig       = plt.figure(figcount)
                        figcount += 1
                        plt.title('Polynomial Coefficients')
                        modal_coeffs = numpy.empty((pE + 1))
                        inter_coeffs = numpy.empty((pE + 1))
                        m_idx        = numpy.empty((pE + 1))
                        for k in range(pE + 1):
                            m_idx[k]        = k
                            modal_coeffs[k] = numpy.log10(abs(modal_poly.node[k]))
                            inter_coeffs[k] = numpy.log10(abs(inter_poly.node[k]))
                        modal_p  = [pX, modal_coeffs[pX] ]
                        modal_2p = [pE, modal_coeffs[pE] ]
                        inter_p  = [pX, inter_coeffs[pX] ]
                        inter_2p = [pE, inter_coeffs[pE] ]

                        plt.scatter(m_idx, modal_coeffs,      color = 'r'     , marker='^', s = 50, label = modal_poly.getType() + ' Coeffs')
                        plt.scatter( modal_p[0],  modal_p[1], color = 'orange', marker='^', s = 55, label = modal_poly.getType() + ' Coeff p')
                        plt.scatter(modal_2p[0], modal_2p[1], color = 'orange', marker='^', s = 60, label = modal_poly.getType() + ' Coeff 2p')

                        plt.scatter(m_idx,   inter_coeffs,    color = 'b'     , marker='o', s = 50, label = inter_poly.getType() + ' Coeffs')
                        plt.scatter( inter_p[0],  inter_p[1], color = 'c'     , marker='o', s = 55, label = inter_poly.getType() + ' Coeff p')
                        plt.scatter(inter_2p[0], inter_2p[1], color = 'c'     , marker='o', s = 60, label = inter_poly.getType() + ' Coeff 2p')

                        plt.legend()
                        plt.xticks(m_idx)

                    # break expansion

                    modal_basis        = modal_poly.getBasis(pE + 1, z, gp) # degree 2 * p => order 2 * p +1
                    inter_basis        = inter_poly.getBasis(pE + 1, z, gp) # degree 2 * p => order 2 * p +1

                    modal_coeff_0_2pm1  = numpy.zeros([2,gp]) # with and without quadrature weights
                    modal_coeff_2p      = numpy.zeros([2,gp])
                    modal_coeff_0_p     = numpy.zeros([2,gp])
                    modal_coeff_p       = numpy.zeros([2,gp])
                    modal_all_coeff     = numpy.zeros([2,gp]) # with and without quadrature weights

                    inter_coeff_0_2pm1 = numpy.zeros([2,gp]) # with and without quadrature weights
                    inter_coeff_2p     = numpy.zeros([2,gp])
                    inter_coeff_0_p    = numpy.zeros([2,gp])
                    inter_coeff_p      = numpy.zeros([2,gp])
                    inter_all_coeff    = numpy.zeros([2,gp]) # with and without quadrature weights
                    check_poly         = numpy.zeros([2,gp])
                    for k in range(gp):
                        for j in range(pE + 1):
                            modj = modal_poly.node[j,0] * modal_basis[k,j]
                            intj = inter_poly.node[j,0] * inter_basis[k,j]
                            modal_all_coeff[0,k] += modj
                            modal_all_coeff[1,k] += modj * w[k]
                            inter_all_coeff[0,k] += intj
                            inter_all_coeff[1,k] += intj * w[k]
                            if j < pE:
                                modal_coeff_0_2pm1[0,k] += modj
                                modal_coeff_0_2pm1[1,k] += modj * w[k]
                                inter_coeff_0_2pm1[0,k] += intj
                                inter_coeff_0_2pm1[1,k] += intj * w[k]
                            else:
                                modal_coeff_2p[0,k] = modj
                                modal_coeff_2p[1,k] = modj * w[k]
                                inter_coeff_2p[0,k] = intj
                                inter_coeff_2p[1,k] = intj * w[k]
                            if j <= pX:
                                modal_coeff_0_p[0,k] += modj
                                modal_coeff_0_p[1,k] += modj * w[k]
                                inter_coeff_0_p[0,k] += intj
                                inter_coeff_0_p[1,k] += intj * w[k]

                        check_poly[0,k] = modal_all_coeff[0,k] - modal_2p_error[k]
                        check_poly[1,k] = inter_all_coeff[0,k] - inter_2p_error[k]
                    # check some things
                    c_mod = check_poly[0].sum(axis=0)
                    c_int = check_poly[1].sum(axis=0)
                    if (c_mod > 1.e-14 or c_int > 1.e-14):
                        print(' !!! Messed up breaking modal expansion: the sum of all modes is not the same than the original projection !!! ')
                        print(' c_mod = ',c_mod, ' c_int = ',c_int)
                    def amplitude(f, w):
                        sum = 0.0
                        Q = len(w)
                        sum_w = 0.0
                        for i in range(Q):
                            sum += f[i] * w[i]
                            sum_w += w[i]
                        return sum / sum_w


                    def signed_log(f):
                        sign = numpy.log10(numpy.abs(f) + 1.e-16)
                        g    = numpy.sign(f) * sign * numpy.log10(numpy.abs(f) + 1.e-16)
                        return g

                    '''aMod02pm1 = amplitude(modal_coeff_0_2pm1[1], w)
                    aMod2p    = amplitude(modal_coeff_2p[1],     w)
                    aMod0p    = amplitude(modal_coeff_0_p[1],    w)
                    aEF       = amplitude(signed_dist_x_alpha, w)
                    print(' AMPLITUDE VALUES',aMod02pm1, aMod2p, aMod0p, aEF)'''

                    #for k in range(gp):
                    #   modal_coeff_0_2pm1[k] = abs(modalSum[k]) + 1.e10
                    #    modal_coeff_2p[k]   = abs(modal_coeff_2p[k]) + 1.e10
                    #    modal_coeff_0_p[k]  = abs(modal_coeff_0_p[k]) + 1.e10
                    #    modal_coeff_0_p[k]  = abs(modal_coeff_0_p[k]) + 1.e10
                #    print(numpy.log10(abs (aMod02pm1)),  numpy.log10(abs (aMod2p)), numpy.log10(abs (aMod0p)), numpy.log10(abs (aEF)))
                #    for k in range(gp):
                #       modal_coeff_0_2pm1[k] = ((modalSum[k]) / abs(aMS  )) * numpy.sign(aMS)   * numpy.log10(abs (aMS))
                #        modal_coeff_2p[k]   = ((modal_coeff_2p[k]) / abs(aRL   )) * numpy.sign(aRL)   * numpy.log10(abs (aRL))
                #        modal_coeff_0_p[k]  = ((modal_coeff_0_p[k]) / abs(aMono )) * numpy.sign(aMono) * numpy.log10(abs (aMono))
                #        modal_coeff_0_p[k]       = ((modal_coeff_0_p[k])      / abs(aEF   )) * numpy.sign(aEF)   * numpy.log10(abs (aEF))
                    '''for k in range(gp):
                        modal_coeff_0_2pm1[1,k]  = signed_log(   modal_coeff_0_2pm1[1,k])
                        modal_coeff_2p[1,k]      = signed_log(       modal_coeff_2p[1,k])
                        modal_coeff_0_p[1,k]     = signed_log(      modal_coeff_0_p[1,k])
                        signed_dist_x_alpha[k] = signed_log(signed_dist_x_alpha[k])
                    '''
                    wi = 0
                    for k in range (2):
                        if k ==0:
                            tit          = 'Modal (Legendre)'
                            coeff_0_2pm1 = modal_coeff_0_2pm1[wi]
                            coeff_0_p    = modal_coeff_0_p[wi]
                            coeff_2p     = modal_coeff_2p[wi]
                            poly_error   = modal_all_coeff[wi]
                        else:
                            tit          = inter_poly.getType()
                            coeff_0_2pm1 = inter_coeff_0_2pm1[wi]
                            coeff_0_p    = inter_coeff_0_p[wi]
                            coeff_2p     = inter_coeff_2p[wi]
                            poly_error   = inter_all_coeff[wi]
                        if showPlots == 2:

                            fig       = plt.figure(figcount)
                            figcount += 1
                            plt.suptitle(tit + ' Representation')
                            ax = plt.subplot(1,4,1)
                            plt.plot(z, coeff_0_p, c = 'b', label = 'Coeffs: 0,...,p')
                            plt.legend(loc='best')

                            ax = plt.subplot(1,4,2)
                            plt.plot(z, coeff_0_2pm1, c = 'b', label = 'Coeffs: 0,...,2p-1')
                            plt.legend(loc='best')

                            ax = plt.subplot(1,4,3)
                            plt.plot(z, coeff_2p, c = 'b', label = 'Coeff: 2p')
                            plt.legend(loc='best')

                            ax = plt.subplot(1,4,4)
                            plt.plot(z,true_error, c = 'orange', linestyle = '-' , label = 'True Error')
                            plt.plot(z,poly_error   , c = 'b', linestyle = '-.',      label = 'All 2p+1 coeffs')
                            plt.plot(z,coeff_2p     , c = 'c', linestyle = '--',      label = 'Coeff 2p')
                            plt.legend(loc='best')

                            fig       = plt.figure(figcount)
                            figcount += 1
                            plt.suptitle(tit + ' Representation')

                            plt.plot(z, coeff_0_p              , c = 'b'     , linestyle = '--', label = 'Coeffs: 0,...,p')
                            plt.plot(z, coeff_0_2pm1           , c = 'c'     , linestyle = '-.', label = 'Coeffs: 0,...,2p-1')
                            plt.plot(z, coeff_2p               , c = 'g'     , linestyle = ':',  label = 'Coeff: 2p')
                            plt.plot(z, true_error, c = 'orange', linestyle = '-', linewidth = 1.25, label = 'True Error')
                            plt.scatter(z,poly_error           , c = 'r'     , marker='*', s = 10, label = 'All 2p+1 coeffs')

                            yzero = [0,0]
                            xzero = [-1,1]
                            plt.plot(xzero, yzero, c = 'gray', linestyle = '-', linewidth = 0.5)
                            plt.legend(loc='best')


                    sumXA = 0.0

                    for j in range (gp):
                        x    [type,i * gp + j] = x_i[j]
                        t    [type,i * gp + j] = t_i[j]
                        alpha[type,i * gp + j] = alpha_i[j]


                        errC [type,i * gp + j] = x_i[j] - alpha_i[j]
                        errT [type,i * gp + j] = true_error[j,0]

                        polC [0,i * gp + j] = xmod_l2gp[j] - alpha_i[j]
                        polC [1,i * gp + j] = xint[j] - alpha_i[j]

                        ep_2p[0,i * gp + j] = modal_coeff_2p[wi, j]
                        ep_2p[1,i * gp + j] = inter_coeff_2p[wi, j]
                        polT [0,i * gp + j] = e_mod_2p_l2gp[j]
                        polT [1,i * gp + j] = e_int_2p_l2gp[j]


                        ea[type] = max ( ea[type    ], abs(true_error[j,0]))
                        derx     = numpy.sqrt(dx_i[j,0] ** 2 + dx_i[j,1] ** 2)
                        sumXA   += true_error[j,0] * true_error[j,0] * w[j] * derx

                    disparity_XA  [type,ref] += sumXA * 0.5
            if (showPlots != 2): figcount = 1
            if ref != 0: continue

            zAXIS  = numpy.zeros(n + 1)
            yAXIS  = numpy.zeros(n + 1)
            ypAXIS = numpy.zeros(n * (pX + 1))
            yuAXIS = numpy.zeros(n * (pT + 1))
            tAXIS  = numpy.zeros([2,n + 1])
            xEP    = numpy.zeros([2,n + 1,dim])
            aEP    = numpy.zeros([2,n + 1,dim])

            for i in range(n):
                zAXIS   [i] = zx      [i * gp][0]
                tAXIS[0][i] = t    [0][i * gp][0]
                tAXIS[1][i] = t    [1][i * gp][0]
                xEP  [0][i] = x    [0][i * gp]
                xEP  [1][i] = x    [1][i * gp]
                aEP  [0][i] = alpha[0][i * gp][0]
                aEP  [1][i] = alpha[1][i * gp][0]
            zAXIS[n]    = zx      [-1][0]
            tAXIS[0][n] = t    [0][-1][0]
            tAXIS[1][n] = t    [1][-1][0]
            xEP[0][n]   = x    [0][-1]
            xEP[1][n]   = x    [1][-1]
            aEP[0][n]   = alpha[0][-1]
            aEP[1][n]   = alpha[1][-1]

            fig       = plt.figure(figcount)
            figcount += 1
            plt.suptitle(' Curves ' + pltInfo)
            '''
            if (dim == 1):
                for type in range(2):
                    plt.subplot(2,3,3 * type + 1)
                    if type == 0: plt.title(' Opti     Solution')
                    else:         plt.title(' Interpol Solution')
                    plt.plot(zx, x[type], c = 'b', linestyle='-.')
                    plt.xlabel('z')
                    plt.subplot(2,3,3 * type + 2)
                    plt.title('Target Solution')
                    plt.plot(zx, alpha[type], c = 'r',     linestyle=':', label = 'alpha(t)')
                    plt.plot(zx, aPF        , c = 'orange', linestyle='-', label = 'alpha(z)')
                    plt.xlabel('z')
                    plt.legend()
                    plt.subplot(2,3,3 * type + 3)
                    plt.title(' Overlap')
                    plt.plot(zx,     aPF    , c = 'orange', linestyle='-',  label = 'alpha(z)')
                    plt.plot(zx,   x[type]  , c = 'b',      linestyle='-.', label = 'x(z)')
                    plt.plot(zx, alpha[type], c = 'r',      linestyle=':',  label = 'alpha(t )')
                    plt.xlabel('z')
                    plt.legend()

            else:
                for type in range(showMesh0):
                    plt.subplot(2,3,3 * type + 1)
                    if type == 0: plt.title(' Opti     Solution')
                    else:         plt.title(' Interpol Solution')
                    plt.plot(x[type,:,0], x[type,:,1], c = 'b', linestyle='-.')
                    plt.xlabel('1st comp')
                    plt.xlabel('2nd comp')

                    plt.subplot(2,3,3 * type + 2)
                    plt.title('Target Solution')
                    plt.plot(alpha[type,:,0], alpha[type,:,1], c = 'r',     linestyle=':', label = 'alpha(t)')
                    plt.plot(      aPF[:,0], aPF[:,1]        , c = 'orange', linestyle='-', label = 'alpha(z)')
                    plt.xlabel('1st comp')
                    plt.xlabel('2nd comp')
                    plt.legend()
                    plt.subplot(2,3,3 * type + 3)
                    plt.title(' Overlap')
                    plt.plot(       aPF[:,0],        aPF[:,1], c = 'orange', linestyle='-',  label = 'alpha(z)')
                    plt.plot(    x[type,:,0],     x[type,:,1], c = 'b',      linestyle='-.', label = 'x(z)')
                    plt.plot(alpha[type,:,0], alpha[type,:,1], c = 'r',      linestyle=':',  label = 'alpha(t )')

                    plt.xlabel('1st comp')
                    plt.xlabel('2nd comp')
                    plt.legend()




            errCN  = numpy.zeros([2, n * gp, dim])
            errTN  = numpy.zeros([2, n * gp])
            poleCN = numpy.zeros([2, n * gp, dim])
            poleTN = numpy.zeros([2, n * gp])

            errCN[0]  =  errC[0] / ea[0]
            errCN[1]  =  errC[1] / ea[1]

            errTN[0]   =  errT[0] / ea[0]
            errTN[1]   =  errT[1] / ea[1]
            polTN[0] = poleT[0] / ea[0]
            polTN[1]  = poleT[1] / ea[1]
            polCN[0]  = poleC[0] / ea[0]
            polCN[1]  = poleC[1] / ea[1]'''


            if dim == 1:
                fig       = plt.figure(figcount)
                figcount += 1
                plt.suptitle('Error from ' + pltInfo)
                for type in range(2):
                    plt.subplot(2,2, 2 * type +1)
                    if type == 0: plt.title('Optimized x')
                    else:         plt.title('Interpol x')
                    plt.plot(zx,  x[type] - aPF, c = 'b', linestyle='-.', label='x(z) - alpha(z)')
                    plt.plot(zAXIS,yAXIS       , c = 'g', linewidth = 0.25, linestyle = ':')
                    plt.xlabel('z')
                    plt.legend()
                    plt.subplot(2,2, 2 * type +2)
                    if type == 0: plt.title('Optimized x and t ')
                    else:         plt.title('Interpol x  and t')
                    plt.plot(zx,errT[type] , c = 'b', linestyle ='-', label='x(z) - alpha(t)')
                    plt.plot(zx,polT[type] , c = 'r', linestyle =':', label='error poly')
                    plt.plot(zx,ep_2p[type], c = 'c', linestyle =':', label='coeff 2p ')

                    plt.plot(zAXIS,yAXIS,    c = 'g', linewidth = 0.25, linestyle = ':')
                    plt.scatter(zAXIS, xEP[type] - aEP[type] , c = 'g', s = 15)
                    plt.xlabel('z')
                    plt.legend()
            else:

                if showPlots == 3:
                    fig       = plt.figure(figcount)
                    figcount += 1
                    for ct in range(2):
                        if ct == 1:
                            curve = alpha[0]
                            leg = ' o t'
                        else:
                            curve = aPF
                            leg = ' '

                        R = 0.5 * (curve[:,0] + curve[:, 1])
                        plt.suptitle(' Showing ' + str(n) + ' Elements for ' + pltInfo)
                        plt.subplot(2,3,3 * ct + 1)
                        plt.plot(zx, R, c = 'b', linestyle='-', label = 'R'+ leg)
                        xep = numpy.zeros(n + 1)
                        for k in range(n): xep[k] = R[k * gp]
                        xep[n] = R[-1]
                        plt.scatter(zAXIS, xep, c = 'r', s = 20)
                        plt.plot(zAXIS, xep, c = 'r', linestyle = '-', linewidth = 0.5)
                        plt.xlabel('z')
                        plt.legend()

                        plt.subplot(2,3,3 * ct + 2)
                        #plt.title(' Promedio, alpha_1, alpha_2')
                        plt.plot(zx, R,          c = 'orange', linestyle=':', label = 'R' + leg, linewidth = 0.5)
                        plt.plot(zx, curve[:,0], c = 'b',      linestyle='-', label = 'alpha_1' + leg)
                        plt.plot(zx, curve[:,1], c = 'c',      linestyle='-', label = 'alpha_2' + leg)

                        plt.scatter(zAXIS, xep, c = 'orange', s = 5)
                        for k in range(n): xep[k] = curve[k * gp,0]
                        xep[n] = curve[-1,0]
                        plt.scatter(zAXIS, xep, c = 'b', s = 5)
                        for k in range(n): xep[k] = curve[k * gp,1]
                        xep[n] = curve[-1,1]
                        plt.scatter(zAXIS, xep, c = 'c', s = 5)
                        plt.xlabel('z')
                        plt.legend()
                        plt.subplot(2,3,3 * ct + 3)

                        #plt.title(' Beta = alpha_1 - R (= R - alpha_2')
                        beta =  curve[:,0] - R
                        plt.plot(zx, beta, c = 'b', linestyle='-', label = 'beta')
                        for k in range(n): xep[k] = beta[k * gp]
                        xep[n] = beta[-1]
                        plt.scatter(zAXIS, xep, c = 'b', s = 5)
                        plt.legend()
                        plt.savefig('average_components_p'+str(pX)+'_n'+str(n)+'.png', bbox_inches='tight')

                    fig       = plt.figure(figcount)
                    figcount += 1
                    Rt        = 0.5 * (alpha[0,:,0] + alpha[0,:,1])
                    R         = 0.5 * (    aPF[:,0] +   aPF[  :,1])
                    betaT     = alpha[0,:,0] - Rt
                    beta      =   aPF[  :,0] - R

                    plt.suptitle(' Showing ' + str(n) + ' Elements for ' + pltInfo)
                    #plt.subplot(1,2,1)
                    plt.plot(zx,    R, c = 'b',      linestyle='--',label = 'R')
                    plt.plot(zx,   Rt, c = 'c',      linestyle='-', label = 'R o t')
                    plt.plot(zx, beta, c = 'r',      linestyle='--',label = 'beta')
                    plt.plot(zx, betaT, c = 'orange',linestyle='-', label = 'beta o t')
                    plt.legend()
                    plt.savefig('average_deviation_p'+str(pX)+'_n'+str(n)+'.png')

                    fig       = plt.figure(figcount)
                    figcount += 1
                    Rx        = 0.5 * (x[0,:,0] + x[0,:,1])
                    betaX     = x[0,:,0] - Rx

                    plt.suptitle(' Showing ' + str(n) + ' Elements for ' + pltInfo)
                    plt.subplot(1,2,1)
                    plt.plot(zx,   Rt, c = 'b',      linestyle='-',label = 'R_x')
                    plt.plot(zx,   Rx, c = 'c',      linestyle='-.', label = 'R o t')
                    plt.plot(zx, betaT, c = 'r',      linestyle='-',label = 'beta_x')
                    plt.plot(zx, betaX, c = 'orange',linestyle='-.', label = 'beta o t')

                    plt.scatter(zAXIS, xep, c = 'r', s = 5)
                    for k in range(n): xep[k] = Rt[k * gp]
                    xep[n] = beta[-1]
                    plt.scatter(zAXIS, xep, c = 'b', s = 5)
                    plt.legend()
                    plt.subplot(1,2,2)
                    plt.plot(zx,Rt - Rx, c = 'c',      linestyle='-',label = 'error (R) ')
                    plt.plot(zx,betaT - betaX, c = 'orange',      linestyle='-.',label = 'error (beta)')
                    plt.scatter(zAXIS, yAXIS, c = 'g', s = 20)
                    plt.plot(zAXIS, yAXIS, c = 'g', linewidth = 0.5)
                    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1e'))

                    plt.legend()
                    plt.savefig('average_x_vs_alpha_p'+str(pX)+'_n'+str(n)+'.png')

                for typ in (range(2)):
                    fig       = plt.figure(figcount)
                    figcount += 1
                    type      = 0
                    if (type == 0): plt.suptitle('Optimal Showing ' + str(n) + ' Elements for ' + pltInfo)
                    else:  plt.suptitle('Initial Showing ' + str(n) + ' Elements for ' + pltInfo)
                    plt.subplot(2,3,1)
                    plt.title(' Solution Curves')
                    plt.plot(x    [type,:,0],     x[type,:,1], c = 'b', linestyle='-', label = 'x(z)')
                    plt.plot(alpha[type,:,0], alpha[type,:,1], c = 'c', linestyle=':',label = 'alpha(t(z))')
                    plt.legend()

                    plt.subplot(2,3,2)
                    plt.title(' 1st Component')
                    plt.plot(zx, x    [type,:,0], c = 'b', linestyle='-', label = 'x(z)')
                    plt.plot(zx, alpha[type,:,0], c = 'c', linestyle=':',label = 'alpha(t(z))')
                    plt.legend()

                    plt.subplot(2,3,3)
                    plt.title(' 2nd Component')
                    plt.plot(zx, x    [type,:,1], c = 'b', linestyle='-', label = 'x(z)')
                    plt.plot(zx, alpha[type,:,1], c = 'c', linestyle=':',label = 'alpha(t(z))')
                    plt.legend()


                    plt.subplot(2,3,4)
                    plt.title(' Total Error Function')
                    plt.plot(zx, errT[type], c = 'g'     , linestyle ='-' , label = 'x(z) - alpha(t(z))')
                    plt.plot(zx, polT[typ], c = 'orange', linestyle ='-.', label = 'error using 2p modal pol ')
                    plt.plot(zx,ep_2p[typ] , c = 'r'     , linestyle =':' , label = 'coeff 2p ')
                    plt.plot   (zAXIS,yAXIS, c = 'g'     , linewidth = 0.25, linestyle = ':')
                    plt.scatter(zAXIS,yAXIS, c = 'g'     , s = 15)
                    plt.xlabel('z')
                    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1e'))
                    plt.legend()


                    plt.subplot(2,3,5)
                    plt.title(' Error 1st Comp')
                    plt.plot(zx, errC[type,:,0], c = 'g'     , linestyle ='-' , label = 'x(z) - alpha(t(z))')
                    plt.plot(zx, polC[typ,:,0], c = 'orange', linestyle =':', label = 'error using 2p modal pol ')


                    plt.plot(zAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
                    plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)
                    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1e'))

                    plt.xlabel('z')
                    plt.legend()


                    plt.subplot(2,3,6)
                    plt.title(' Error 2nd Comp')
                    plt.plot(zx, errC[type,:,1], c = 'g'     , linestyle ='-.', label = 'x(z) - alpha(t(z))')
                    plt.plot(zx, polC[typ,:,1], c = 'orange', linestyle =':' , label = 'error using 2p modal pol ')
                    plt.plot(   zAXIS,yAXIS,     c = 'g'     , linewidth = 0.25, linestyle = ':')
                    plt.scatter(zAXIS,yAXIS,      c = 'g'    , s = 15)
                    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1e'))

                    plt.xlabel('z')
                    plt.legend()


                '''
                fig       = plt.figure(figcount)
                figcount += 1
                for type in range(showMesh0):

                    plt.suptitle('Error from ' + pltInfo)
                    plt.subplot(2,3, 3 * type +1)
                    if type == 0: plt.title('1st comp, x Opt')
                    else:         plt.title('1st comp, x Interp')
                    plt.plot(zx,errC[type,:,0], c = 'b', linestyle='-.', label='x(z) - alpha (t)')
                    plt.plot(zx,polC[type,:,0], c = 'r', linestyle =':', label='error poly')
                    plt.plot(zAXIS,yAXIS      , c = 'g', linewidth = 0.25, linestyle = ':')
                    plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)

                    plt.xlabel('z')
                    plt.legend()
                    plt.subplot(2,3, 3 * type +2)
                    if type == 0: plt.title('2nd comp, x Opt')
                    else:         plt.title('2nd comp, x Interp')
                    plt.plot(zx,errC[type,:,1], c = 'c', linestyle='-.',label='x(z) - alpha (t)')
                    plt.plot(zx,polC[type,:,1], c = 'r', linestyle =':',label='error poly')
                    plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)

                    plt.plot(zAXIS,yAXIS       , c = 'g', linewidth = 0.25, linestyle = ':')
                    plt.xlabel('z')
                    plt.legend()
                    plt.subplot(2,3, 3 * type +3)
                    if type == 0: plt.title('Signed Error x Opt')
                    else:         plt.title('Signed Error x Interp')
                    plt.plot(zx,errT[type], c = 'r', linestyle='-', label = 'x(t) - alpha(t)')
                    plt.plot(zx,polT[type], c = 'orange', linestyle='-.', label = 'error poly')
                    plt.plot(zx,ep_2p[type], c = 'c', linestyle =':', label='coeff 2p ')

                    plt.plot(zAXIS,yAXIS    , c = 'g', linewidth = 0.25, linestyle = ':')
                    plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)

                    plt.xlabel('z')
                    plt.legend()

                fig       = plt.figure(figcount)
                figcount += 1
                plt.suptitle('Normalized Error distance ' + pltInfo)

                for type in range(2):
                    plt.subplot(2,3,3 * type + 1)
                    if type == 0: plt.title('Optimized 1st comp')
                    else:         plt.title('Interpol  1st comp')
                    plt.plot(zx, errCN[type,:,0] - poleCN[type,:,0], c = 'b'  , linestyle='-')
                    plt.plot(zAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
                    plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)

                    plt.subplot(2,3,3 * type + 2)
                    if type == 0: plt.title('Optimized 2nd comp')
                    else:         plt.title('Interpol  2nd comp')
                    plt.plot(zx, errCN[type,:,1] - poleCN[type,:,1], c = 'b'  , linestyle='-', label='1st comp')
                    plt.plot(zAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
                    plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)

                    plt.subplot(2,3,3 * type + 3)
                    if type == 0: plt.title('Optimized dist')
                    else:         plt.title('Interpol  dist')
                    plt.plot(zx,   errT[type] - poleT[type], c = 'g'      , linestyle='-')
                    plt.plot(zAXIS,yAXIS, c = 'g', linewidth = 0.25, linestyle = ':')
                    plt.scatter(zAXIS,yAXIS,  c = 'g', s = 15)
            '''
        for type in range(2):
            if (type == 0): print(' ********  OPTIMIZED     MESH ***********')
            else:           print(' ********  INTERPOLATING MESH ***********')
            convergence_IO(nR, ne, disparity_e[type]   , pX, pT, 'ELOI       DISPARITY: || x_p (xi) - alpha (xi)||_sigma')
            convergence_IO(nR, ne, disparity_XA[type]  , pX, pT, 'MY         DISPARITY: || x_p (xi) - alpha (xi)||_sigme')
        if (showPlots >= 1): plt.show()









if __name__ == '__main__':

    argc = len(sys.argv)
    if argc != 8:
        print (" I NEED dimension + degree x  + degree t + initial elements + refinements + cure type + show plots")
        print(sys.argv)
        quit(1)
    dim       = int(sys.argv[1])  # number of elements
    degX      = int(sys.argv[2])  # number of elements
    degT      = int(sys.argv[3])  # number of elements
    elmts     = int(sys.argv[4])  # number of elements
    refine    = int(sys.argv[5])  # number of elements
    curve     = int(sys.argv[6])  # number of elements
    showPlots = int(sys.argv[7])  # number of elements
    print(' SPACE DIMENSIONS ',dim)
    I = [0,1]
    if dim == 2:
        if ( curve == 0):
            I = [0,numpy.pi]
            print(" SOLVING alpha = (cos(x), sin(x)) x in [0, pi]")
        elif ( curve == 10):
            I = [0,2 * numpy.pi]
            print(" SOLVING alpha = (cos(x), sin(x)) x in [0, 2pi]")
    else:
        if   (curve ==  0):
            I = [0, numpy.pi]
            print(" SOLVING COS(x) x in [0, pi]")
        elif (curve == 10):
            I = [0, 2.0 * numpy.pi]
            print(" SOLVING COS(x) x in [0, 2pi]")
        elif (curve == 5):
            I = [1, 2]
            print(" SOLVING a poly deg 5 ")
    TestDistanceFunctionOptimization.testDistanceFunction(dim, degX, degT, elmts, refine, curve, I, showPlots)
