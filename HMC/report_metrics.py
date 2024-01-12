#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: meysamhashemi INS Marseille

"""
import os
import sys
import numpy as np
import re

import numpy as np
from scipy.spatial.distance import pdist, squareform

#####################################################
def LSE(x1, x2):
    return np.sum((x1 - x2)**2)
#####################################################
def Err(x1, x2):
    return np.sum(np.abs(x1 - x2))
#####################################################    
def RMSE(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).mean()) 
#####################################################
def LSE_obs(Obs, Obs_lo, Obs_hi):
    return np.average([LSE(Obs, Obs_lo), LSE(Obs, Obs_hi)])
#####################################################
def z_score(true_mean, post_mean, post_std):
    return np.abs((post_mean - true_mean) / post_std)
#####################################################
def shrinkage(prior_std, post_std):
    return 1 - (post_std / prior_std)**2
#####################################################
def Rmse_metric(X, Y):
    N=len(X.flatten())
    dS = (X.flatten() - Y.flatten())
    Rmse_value = np.sqrt((1/N)*dS.dot(dS))
    return Rmse_value
#####################################################
def canberra_metric(X, Y):
    N=len(X.flatten())
    d = np.abs(X.flatten() - Y.flatten()) / (np.abs(X.flatten()) + np.abs(Y.flatten()))
    canberra_value = np.nansum(d)/N
    return canberra_value 
#####################################################
def dvar(X):
    """Computes the distance variance of a matrix X.
    """
    return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))
#####################################################
def cent_dist(X):
    """Computes the pairwise euclidean distance between rows of X and centers
     each cell of the distance matrix with row mean, column mean, and grand mean.
    """
    M = squareform(pdist(X))    # distance matrix
    rmean = M.mean(axis=1)
    cmean = M.mean(axis=0)
    gmean = rmean.mean()
    R = np.tile(rmean, (M.shape[0], 1)).transpose()
    C = np.tile(cmean, (M.shape[1], 1))
    G = np.tile(gmean, M.shape)
    CM = M - R - C + G
    return CM
#####################################################
def dcov(X, Y):
    """Computes the distance covariance between matrices X and Y.
    """
    n = X.shape[0]
    XY = np.multiply(X, Y)
    cov = np.sqrt(XY.sum()) / n
    return cov
#####################################################
def dcor(X, Y):
    """Computes the distance correlation between two matrices X and Y.
    X and Y must have the same number of rows.
    >>> X = np.matrix('1;2;3;4;5')
    >>> Y = np.matrix('1;2;9;4;4')
    >>> dcor(X, Y)
    0.76267624241686649
    """
    assert X.shape[0] == Y.shape[0]

    A = cent_dist(X)
    B = cent_dist(Y)

    dcov_AB = dcov(A, B)
    dvar_A = dvar(A)
    dvar_B = dvar(B)

    dcor = 0.0
    if dvar_A > 0.0 and dvar_B > 0.0:
        dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)

    return dcor
#####################################################
#####################################################
