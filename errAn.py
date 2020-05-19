import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy
import math
import parula
import pylab

import vector_functions as vf

eScreen = 1
eEPS    = 0

FS = 20
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


class mypalette:
    def __init__(self,nC, showColors=False):
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
                  self.cmap[12],self.cmap[16] ]
        if (showColors == True): print(self.c)
    #    self.lw = [2.5,2.5,2.5,2.5,2.5]
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


def fig_params1(xax, yax, x, y, subdiv, xl):

    xt = [min(x), max(x)]
    yt = [min(y), max(y)]
    fp (xax, yax, xt, yt, subdiv, xl)


def fig_params(xax, yax, x, y, x2, y2, subdiv, xl):

    xt = [min(min(x), min(x2)), max(max(x), max(x2))]
    yt = [min(min(y), min(y2)), max(max(y), max(y2))]
    fp (xax, yax, xt, yt, subdiv, xl)


def fig_params3(xax, yax, x, y, x2, y2, x3, y3, subdiv, xl):

    xt = [min(min(x), min(x2)), max(max(x), max(x2))]
    yt = [min(min(y), min(y2)), max(max(y), max(y2))]
    xt = [min(xt[0], min(x3)), max(xt[1], max(x3))]
    yt = [min(yt[0], min(y3)), max(yt[1], max(y3))]
    fp (xax, yax, xt, yt, subdiv, xl)

def fp (xax, yax, xt, yt, subdiv, xl):
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

def fig_header(count, visType):
    #plt.close()
    fig = plt.figure(count)
    if visType == eEPS:
        fig = plt.rc('text', usetex=True)
        fig = plt.rc('font', family='serif')
        fig = plt.rc('font',size=FS)
        fig = plt.tick_params(which='both',      # both major and minor ticks are affected
                            top=False)
        fig = plt.figure(frameon=False)
    return count + 1


def fig_leg(root, n):
    ax        = pylab.gca()
    figlegend = pylab.figure(figsize=(3,2))
    # produce a legend for the objects in the other figure
    pylab.figlegend(*ax.get_legend_handles_labels(), ncol = n,  loc = 'center', frameon = False)
    os = root + 'legend.eps'
    figlegend.savefig(os, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_modes(xp, figcount, rows, poly, pX, pID, eID, name, visType):
    size  = 2
    last  = poly.n - 1
    print(' PLOT MODES FOR PT ', pID,' FIG  COUNT ', figcount)
    if eID == 0:
        fig_header(figcount, visType)
    if visType == eScreen:
        plt.subplot(rows, 1, pID + 1)
    i_nor = numpy.empty((poly.n - size))
    c_nor = numpy.empty((poly.n - size))

    i_spe = numpy.empty((size))
    c_spe = numpy.empty((size))
    ik     = 0
    jk     = 0
    for k in range(poly.n):
        dk = vf.xnorm(poly.dim, 1, poly.node[k])
        if k == pX or k == last:
            i_spe[ik] = k
            c_spe[ik] = numpy.log10(dk)
            ik        += 1
        else:
            i_nor[jk]  = k
            c_nor[jk]  = numpy.log10(dk)
            jk        += 1
    if eID == 0:
        idx = numpy.append(i_nor, i_spe)
        coe = numpy.append(c_nor, c_spe)
        fig_params('%1d', '%1.1f', idx, coe, idx, coe, poly.n, 'Coefficients')
        plt.xticks(idx)
    plt.scatter(i_nor, c_nor, color = xp.c[0], marker='o', s = 25)
    plt.scatter(i_spe, c_spe, color = xp.c[1], marker='*', s = 50)
    if visType == eEPS:
        plt.savefig('modes_' + name + '.eps', bbox_inches='tight', pad_inches=0)
        plt.close()


def myplot(x, y, mp, j, name, zAXIS= None, yAXIS = None, plot_axis = False):
    plt.plot(x, y, ls = mp.ls[j], linewidth= mp.lw[j], color = mp.c[j], label = name)
    if (plot_axis == True):
        plt.plot   (zAXIS,yAXIS, ls = ':' ,lw = 0.5, color = mp.ax)
        plt.scatter(zAXIS,yAXIS,  s = 10,  color = mp.ax)


def plot_basis(xp, f_poly, np , pt, name, z, basis, p):
    fig_header(f_poly, 1)
    plt.subplot(np ,1, pt)
    plt.title( name + ' Basis')
    print(' max leg pol ', p)
    for j in range (p):
        jj = j * 2 + 1
        jj = jj % (xp.totColors)
        plt.plot(z, basis[:,j],  color = xp.cmap[jj], label =r'$L_' + str(j)+'$')
    fig_params('%1.1f', '%1.1f',z, [-1,1], z, [-1,1],2, '$\\xi$')

def compare_l2_projection(fcount, name, xp, dim, z, u, uh):
    fcount = fig_header(fcount, 1)
    print('compare l2 projection --> fcount ', fcount - 1)
    plt.title(' Error for ' + name)
    plt.subplot(1,2,1)
    if dim == eScreen:
        myplot(z, u, xp,  0, 'Original')
        myplot(z, uh, xp,  1,  'Reconstruction')
        fig_params('%1.1f', '%1.1f', z, u[:,0], z, uh[:,0])
    else:
        myplot( u[:,0],  u[:,1], xp, 0,  'Original')
        myplot(uh[:,0], uh[:,1], xp, 1,  'Reconstruction')
        fig_params('%1.1f', '%1.1f', u[:,0], u[:,1], uh[:,0], uh[:,1], 3, '$\\xi$')
    plt.subplot(1,2,2)

    aux = vf.xnorm(dim, len(z), u - uh)
    myplot(z, aux, xp, 2, 'Error')
    fig_params('%1.1f', '%1.1e',z, aux, z, aux, 3, '$\\xi$')
    return fcount

def plot_curve(fcount, n, pltInfo, xp, dim, z, c1, name1, c2, name2):
    fcount = fig_header(fcount, 1)
    print(' plot_curve --> fcount ', fcount - 1)
    plt.suptitle(' Solution Curves showing ' + str(n) + ' Elements for ' + pltInfo)
    if dim == eScreen:
        myplot(z, c1[:,0], xp, 0, name1)
        myplot(z, c2[:,0], xp, 1, name2)
        fig_params('%1.1f', '%1.1f',z, c1[:,0], z, c2[:,0], 3, '$\\xi$')
    else:
        plt.subplot(1,3,1)
        plt.title(' 2D Curves')
        myplot(c1[:,0], c1[:,1], xp, 0, name1)
        myplot(c2[:,0], c2[:,1], xp, 1, name2)
        fig_params('%1.1f', '%1.1f',c1[:,0], c1[:,1], c2[:,0], c2[:,1], 3, '$\\xi$')

        plt.subplot(1,3,2)
        plt.title(' 1st Component')
        myplot(z, c1[:,0],xp, 0, name1)
        myplot(z, c2[:,0],xp, 1, name2)
        fig_params('%1.1f', '%1.1f',z, c1[:,0], z, c2[:,0], 3, '$\\xi$')

        plt.subplot(1,3,3)
        plt.title(' 2nd Component')
        myplot(z, c1[:,1],xp, 0, name1)
        myplot(z, c2[:,1],xp, 1, name2)
        fig_params('%1.1f', '%1.1f',z, c1[:,1], z, c2[:,1], 3, '$\\xi$')
    return fcount

def break_expansion(dim, f, gp, order, p, q):
    f0 = numpy.zeros((gp, dim))
    f1 = numpy.zeros((gp, dim))
    f2 = numpy.zeros((gp, dim))
    for j in range(gp):
        for i in range(order):
            if   i < p: f0[j] += f[i,j]
            elif i < q: f1[j] += f[i,j]
            else      : f2[j] += f[i,j]
    return f0, f1, f2








def error_plots(dim, fcount, header, visType, showLeg, xp, np, pt, pE, pX, z, xAXIS, yAXIS, exact, poly, exp_by_mode):

    size = len(z)
    for d in range(dim):
        if dim == eScreen:
            ex = exact
            po = poly
            e0p,  ep2pm1,  e2p  = break_expansion(1, exp_by_mode, size, pE + 1, pX + 1, pE)
            na = ' '
        else:
            p0, p1, p2 = break_expansion(dim, exp_by_mode, size, pE + 1, pX + 1, pE)
            e0p    = p0[:,d]
            ep2pm1 = p1[:,d]
            e2p    = p2[:,d]
            ex     = exact[:,d]
            po     = poly [:,d]
            na = '_comp_' + str(d)

        name = header + na
        lp0 = 'coeffs 0,..,' + str(pX)
        if (pX == 2):
            lp1 = 'coeff ' + str(pE - 1)
        elif (pX == 3):
            lp1 = 'coeffs '+ str(pX + 1) + ',' + str(pE - 1)
        else:
            lp1 = 'coeffs '+ str(pX + 1) + '...' + str(pE - 1)
        lp2 = 'coeff '  + str(pE)

        fcount = fig_header(fcount, visType)
        print(' error plot --> fcount ', fcount - 1)

        if (visType == eScreen):
            plt.suptitle(name)
            plt.subplot(2, 2, 1)
        myplot(z, ex, xp, 0, '$exact$')
        myplot(z, po, xp, 1, '$poly$')

        myplot(z, e2p    , xp, 2, lp2, xAXIS, yAXIS, True)
        myplot(z, ep2pm1, xp, 3, lp1)
        myplot(z, e0p    , xp, 4, lp0)
        fig_params('%1.1f', '%1.1e',z, ex, z, po, 2, '$\\xi$')

        if visType == eEPS:
            of = name + 'Overlap.eps'
            plt.savefig(of, bbox_inches='tight', pad_inches=0)
            if (showLeg == True):
                fig_leg(name + 'Overlap', 5)
            plt.close()

        elif showLeg == True:
            plt.legend(loc='upper center', bbox_to_anchor=(1, 1.15), ncol = 5)

        if (visType == eScreen): plt.subplot(2, 2, 2)
        myplot(z, ex , xp,  0, 'exact', xAXIS, yAXIS, True)
        myplot(z, e2p, xp,  2, lp2, xAXIS, yAXIS, True)
        fig_params('%1.1f', '%1.1e',z,e2p, z,ex, 2 , '$\\xi$')

        if visType == eEPS:
            of = name + '2p.eps'
            plt.savefig(of, bbox_inches='tight', pad_inches=0)
            plt.close()

        if (visType == eScreen): plt.subplot(2, 2, 3)
        myplot(z, ep2pm1 , xp,  3, lp1, xAXIS, yAXIS, True)
        fig_params('%1.1f', '%1.1e',z,ep2pm1, z,ep2pm1, 2 , '$\\xi$')

        if visType == eEPS:
            of = name + 'p_2pm1.eps'
            plt.savefig(of, bbox_inches='tight', pad_inches=0)
            plt.close()

        if (visType == eScreen): plt.subplot(2, 2, 4)
        myplot(z, e0p , xp,  4, lp0, xAXIS, yAXIS, True)
        fig_params('%1.1f', '%1.1e',z, e0p, z,e0p, 2 , '$\\xi$')

        if visType == eEPS:
            of = name + '0p.eps'
            plt.savefig(of, bbox_inches='tight', pad_inches=0)
            plt.close()
    return fcount


def error_decomp_default(dim, fcount, xp, plt_tit, visType,  pE, z, total, \
exp_by_mode, total_dim, exp_by_mode_dim,  xAXIS, yAXIS):
    title = plot_tit+' Error decomp. '
    error_decomp(dim, fcount, xp, plt_tit, visType,  pE, z, total, \
        exp_by_mode, total_dim, exp_by_mode_dim,  xAXIS, yAXIS, title)

def error_decomp(dim, fcount, xp, plt_tit, visType,  pE, z, total, \
exp_by_mode, total_dim, exp_by_mode_dim,  xAXIS, yAXIS, title):
    mat    = pE + 2
    fcount = fig_header(fcount, visType)
    if visType == eScreen:
        plt.suptitle(title)
    if (mat % 2 != 0):
        half = int( (mat + 1) / 2)
        col  = half
        row  = mat / half - 1
    else:
        half = int(mat / 2)
        col  = half
        row  = mat / half
        for j in range (mat):
            if j == pE + 1:
                name = 'All'
                fun  = total
                if dim != 1:
                    fun_2 = total_dim
            else:
                name = 'L_' + str(j)
                fun  = exp_by_mode[j]
                if dim != 1:
                    fun_2 = exp_by_mode_dim[j]
            if visType == eScreen:
                plt.subplot(row, col , j + 1)
                plt.title(name)
            myplot(z, fun, xp,  0, 'abs', xAXIS, yAXIS, True)
            if dim != 1:
                for d in range(dim):
                    myplot(z, fun_2[:,d], xp, 1 + d,  'comp_' + str(d))
                fig_params3('%1.1f', '%1.e',z, fun_2[:,0], z, fun_2[:,1], z, fun, 2,    '$\\xi$')
            else: fig_params('%1.1f', '%1.e',z, fun, z, fun, 2, '$\\xi$')
            #plt.plot   (xAXIS,yAXIS, ls = ':' ,lw = 0.5, color = xp.ax)
            #plt.scatter(xAXIS,yAXIS,  s = 10,  color = xp.ax)
            plt.xticks([], [])
            plt.xlabel(None)
            if visType == eEPS:
                plt.savefig(name + '.eps', bbox_inches='tight', pad_inches=0)
                plt.close()
            elif j == pE + 1:
                plt.legend(loc='lower center', bbox_to_anchor=(-1.25, -0.25), ncol = 3)
    return (fcount +1 )

def error_decomp_bis(dim, fcount, xp, plt_tit, visType,  pE, z, total, exp_by_mode, total_dim, exp_by_mode_dim,  xAXIS, yAXIS  ):
    mat    = pE + 2
    if dim == eScreen: return
    else:
        for d in range(dim + 1):
            fcount = fig_header(fcount, visType)
            if visType == eScreen:
                if (d == dim):
                    plt.suptitle(plt_tit + 'absolute error decomposition')
                else:
                    plt.suptitle(plt_tit + 'error decomposition: coordinate ' + str (d))


            if (mat % 2 != 0):
                half = int( (mat + 1) / 2)
                col  = half
                row  = mat / half - 1
            else:
                half = int(mat / 2)
                col  = half
                row  = mat / half
            for j in range (mat):
                if j == pE + 1:
                    name = 'All'
                    if d == (dim):
                        fun  = total
                    else: fun  = total_dim[:,d]
                else:
                    name = 'L_' + str(j)
                    if d == (dim):
                        fun  = exp_by_mode[j]
                    else:
                        print(exp_by_mode_dim)
                        fun  = exp_by_mode_dim[j,:,d]
                if visType == eScreen:
                    plt.subplot(row, col , j + 1)
                    plt.title(name)
                if d == dim:
                    myplot(z, fun, xp,  0, 'Absolute', xAXIS, yAXIS, True)
                else:
                    myplot(z, fun, xp,  0, 'comp_' + str(d), xAXIS, yAXIS, True)
                fig_params('%1.1f', '%1.e',z, fun, z, fun, 2, '$\\xi$')
                #plt.plot   (xAXIS,yAXIS, ls = ':' ,lw = 0.5, color = xp.ax)
                #plt.scatter(xAXIS,yAXIS,  s = 10,  color = xp.ax)
                plt.xticks([], [])
                plt.xlabel(None)
                if visType == eEPS:
                    plt.savefig(name + '.eps', bbox_inches='tight', pad_inches=0)
                    plt.close()
                elif j == pE + 1:
                    plt.legend(loc='lower center', bbox_to_anchor=(-1.25, -0.25), ncol = 3)
    return (fcount +1 )



def compare_errors_tnb (fcount, visType, xp, z,  et, en, ed ):

    fcount = fig_header(fcount, visType)
    if (visType == eScreen):    plt.subplot(1,3,1)
    myplot(z,  ed, xp, 0, 'Absolute')
    myplot(z,  et, xp, 1, 'Tangent')
    myplot(z,  en, xp, 2, 'Normal')
    fig_params3('%1.1f', '%1.e',z, et, z, ed, z, en, 2, '$\\xi$')

    if visType == eScreen:  plt.legend()
    else:
        plt.xticks([], [])
        plt.xlabel(None)
        plt.savefig('error_tnb' + '.eps', bbox_inches='tight', pad_inches=0)

        plt.close()

    if (visType == eScreen):    plt.subplot(1,3,2)

    myplot(z, ed - et, xp, 1, 'Tangent')
    fig_params1('%1.1f', '%1.e',z, ed - et,  2, '$\\xi$')
    if visType == eScreen:  plt.legend()
    else:
        plt.xticks([], [])
        plt.xlabel(None)
        plt.savefig('error_tan_vs_abs' + '.eps', bbox_inches='tight', pad_inches=0)
        plt.close()

    if (visType == eScreen):    plt.subplot(1,3,3)

    myplot(z, ed - en, xp, 2, 'Normal')
    fig_params1('%1.1f', '%1.e',z, ed - en,  2, '$\\xi$')
    if visType == eScreen:  plt.legend()
    else:
        plt.xticks([], [])
        plt.xlabel(None)
        plt.savefig('error_nor_vs_abs' + '.eps', bbox_inches='tight', pad_inches=0)
        plt.close()
    return fcount + 1
