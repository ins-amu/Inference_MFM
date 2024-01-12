#!/usr/bin/env python3
"""
@author: meysamhashemi  INS Marseille

"""

import os
import sys
import time

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.optimize import root

def Montbrio_model(v0, r0, delta, eta, J, I_input, dt, sigma):
    
    nsteps=len(I_input)
    r, v= np.zeros(nsteps), np.zeros(nsteps)
    
    v_init, r_init=v0, r0
    v[0],r[0]=v_init, r_init
    
    for i in range(1, nsteps):
        dr =(delta/np.pi) +2*r[i-1]*v[i-1]
        dv = v[i-1]**2  -(np.pi*r[i-1])**2 + J*r[i-1]+ eta +I_input[i-1]
        r[i]=(r[i-1] + dt * dr +np.sqrt(dt) *sigma * np.random.randn())
        v[i]=(v[i-1] + dt * dv +np.sqrt(dt) *sigma * np.random.randn())    
    
    return v,r


def plot_nullcline(ax, delta, eta, J, I0, linewidth=3, style='--',  vmin=-20,vmax=20):
    v = np.linspace(vmin,vmax,100000)
    nullcline_v=-delta/(2*np.pi*v)
    nullcline_r=v**2  -(np.pi*(-delta/(2*np.pi*v)))**2 + J*(-delta/(2*np.pi*v))+ eta +I0
    ax.plot(v, nullcline_v , style, color='khaki', linewidth=linewidth, label='v-nullcline')
    ax.plot(v, nullcline_r , style, color='y', linewidth=linewidth, label='r-nullcline')
    #ax.legend(fontsize=14, frameon=False)


def MontbrioFlow(x, t, delta, eta, J, I0):
    Fr=(delta/np.pi) +(2*x[0]*x[1])
    Fv=(x[1]**2)  -(np.pi*x[0])**2 + J*x[0]+ eta +I0
    F=np.array([Fr, Fv])
    return F

def plot_vector_field(ax, param, xrange, yrange, steps=1000):
    x = np.linspace(xrange[0], xrange[1], steps)
    y = np.linspace(yrange[0], yrange[1], steps)
    X,Y = np.meshgrid(x,y)
    dx,dy = MontbrioFlow([X,Y],0,**param)   
    ax.streamplot(X,Y,dx, dy, density=2.0, color='lightgray')
    ax.contour(X,Y,dx, [0], linestyles='--', linewidths=3, colors="khaki", alpha=0.9, zorder=4)
    ax.contour(X,Y,dy, [0], linestyles='--', linewidths=3, colors="y", alpha=0.9, zorder=4)
    ax.set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))
    ax.plot([], [], 'khaki', linestyle='--', linewidth=2, label='v-nullcline')
    ax.plot([], [], 'y', linestyle='--', linewidth=2, label='r-nullcline')
    ax.legend(loc='lower right', frameon=False, fontsize=14)    


def find_roots(delta, eta, J, I0):
    coef = [1, 0, (eta+I0), -(J*delta)/(2*np.pi), -(delta/2)**2]
    # We are only interested in real roots.
    # np.isreal(x) returns True only if x is real. 
    # The following line filter the list returned by np.roots
    # and only keep the real values. 
    roots = [np.real(v) for v in np.roots(coef) if np.isreal(v)]
    # We store the position of the equilibrium (r*,v*). 
    return [[-delta/(2*np.pi*v), v] for v in roots]


EQUILIBRIUM_COLOR = {'Stable node':'C6',
                    'Unstable node':'C1', 
                    'Saddle':'C2',
                    'Stable focus':'C3',
                    'Unstable focus':'C4',
                    'Center':'C0'}

def Jacobian_Montbrio(r, v, delta, eta, J, I0):
    """ Jacobian matrix of the Monbrio's excitable system
    Args
    ====
    r, v, delta, eta, J
    Return: np.array 2x2"""
    return np.array([[2*v, 2*r],
                       [((-2*(np.pi**2)*r)+J), 2*v]])


# import sympy 
# sympy.init_printing()

# # Define variable as symbols for sympy
# r, v = sympy.symbols("r, v")
# delta, eta, J, I0 = sympy.symbols("delta, eta, J, I0")

# # Symbolic expression of the system
# drdt = 2*r*v +(delta/np.pi)
# dvdt = v**2 - (np.pi*r)**2 + J*r + eta+I0

# # Symbolic expression of the matrix
# sys = sympy.Matrix([drdt, dvdt])
# var = sympy.Matrix([r, v])
# jac = sys.jacobian(var)

# # You can convert jac to a function:
# jacobian_Montbrio_symbolic = sympy.lambdify((r, v, delta, eta, J, I0), jac, dummify=False)

# #jacobian_Montbrio = jacobian_Montbrio_symbolic
# jac


def stability(jacobian):
    """ Stability of the equilibrium given its associated 2x2 jacobian matrix. 
    Use the eigenvalues. 
    Args:
        jacobian (np.array 2x2): the jacobian matrix at the equilibrium point.
    Return:
        (string) status of equilibrium point.
    """
    
    eigv = np.linalg.eigvals(jacobian)
    
    
    if all(np.real(eigv)==0) and all(np.imag(eigv)!=0):
        nature = "Center" 
    elif np.real(eigv)[0]*np.real(eigv)[1]<0:
        nature = "Saddle"
    else: 
        stability = 'Unstable' if all(np.real(eigv)>0) else 'Stable'
        nature = stability + (' focus' if all(np.imag(eigv)!=0) else ' node')
    return nature

def stability_alt(jacobian):
    """ Stability of the equilibrium given its associated 2x2 jacobian matrix. 
    Use the trace and determinant. 
    Args:
        jacobian (np.array 2x2): the jacobian matrix at the equilibrium point.
    Return:
        (string) status of equilibrium point.
    """
    
    determinant = np.linalg.det(jacobian)
    trace = np.matrix.trace(jacobian)
    if np.isclose(trace, 0):
        nature = "Center (Hopf)"
    elif np.isclose(determinant, 0):
        nature = "Transcritical (Saddle-Node)"
    elif determinant < 0:
        nature = "Saddle"
    else:
        nature = "Stable" if trace < 0 else "Unstable"
        nature += " focus" if (trace**2 - 4 * determinant) < 0 else " node"
    return nature


import matplotlib.patches as mpatches #used to write custom legends


def plot_phase_diagram(v0, r0, param, I_input, dt, sigma, ax=None, title=None):
    """Plot a complete Montbrio model phase Diagram in ax.
    Including isoclines, flow vector field, equilibria and their stability"""
    if ax is None:
        ax = plt.gca()
    #if title is None:
        #title = "Phase space, {}".format(param) 
    
    ax.set(xlabel='r', ylabel='v', title=title)
        
    xrange = (-1, 4) 
    yrange =(-5, 5)
    v,r = Montbrio_model(v0, r0, param['delta'], param['eta'], param['J'], I_input, dt, sigma)
    plot_vector_field(ax, param, xrange, yrange)
    #plot_nullcline(ax, **param)
 
    
    # Plot the equilibria 
    eqnproot = find_roots(**param)
    eqstability = [stability(Jacobian_Montbrio(e[0],e[1], **param)) for e in eqnproot] 
    for e,n in zip(eqnproot,eqstability):
        ax.scatter(*e, color=EQUILIBRIUM_COLOR[n], s=120, zorder=8)
        
        ax.plot(r,v,  marker="o", markersize=0, lw=2, color='blue', alpha=0.5, label='Trajectory', zorder=4)
        ax.plot(r[0],v[0],  marker="o", markersize=10, lw=0, color='blue', alpha=0.5, zorder=5)
    

    # Legend
    labels = frozenset(eqstability)
    ax.legend([mpatches.Patch(color=EQUILIBRIUM_COLOR[n]) for n in labels], labels, 
           loc='lower right', fontsize=12)
