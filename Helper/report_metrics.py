#!/usr/bin/env python3
"""
@author: meysamhashemi INS Marseille

"""
import os
import sys

import numpy as np
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
import torch

from scipy.stats import gaussian_kde
from sbi.utils.plot import _get_default_opts, _update, ensure_numpy





def _get_limits(samples, limits=None):

    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Dimensionality of the problem.
    dim = samples[0].shape[1]

    if limits == [] or limits is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(limits) == 1:
            limits = [limits[0] for _ in range(dim)]
        else:
            limits = limits
    limits = torch.as_tensor(limits)

    return limits


def posterior_peaks(samples, return_dict=False, **kwargs):
    '''
    Finds the peaks of the posterior distribution.

    Args:
        samples: torch.tensor, samples from posterior
    Returns: torch.tensor, peaks of the posterior distribution
            if labels provided as a list of strings, and return_dict is True
            returns a dictionary of peaks

    '''

    opts = _get_default_opts()
    opts = _update(opts, kwargs)

    limits = _get_limits(samples)
    samples = samples.numpy()
    n, dim = samples.shape

    try:
        labels = opts['labels']
    except:
        labels = range(dim)

    peaks = {}
    if labels is None:
        labels = range(dim)
    for i in range(dim):
        peaks[labels[i]] = 0

    for row in range(dim):
        density = gaussian_kde(
            samples[:, row],
            bw_method=opts["kde_diag"]["bw_method"])
        xs = np.linspace(
            limits[row, 0], limits[row, 1],
            opts["kde_diag"]["bins"])
        ys = density(xs)

        # y, x = np.histogram(samples[:, row], bins=bins)
        peaks[labels[row]] = xs[ys.argmax()]

    if return_dict:
        return peaks
    else:
        return list(peaks.values())




#########################################################
from scipy.spatial.distance import pdist, squareform

from scipy.stats import wasserstein_distance, energy_distance, ks_2samp

from scipy.special import rel_entr

from sklearn.metrics import mutual_info_score



def dvar(X):
    """Computes the distance variance of a matrix X.
    """
    return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))


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


def dcov(X, Y):
    """Computes the distance covariance between matrices X and Y.
    """
    n = X.shape[0]
    XY = np.multiply(X, Y)
    cov = np.sqrt(XY.sum()) / n
    return cov



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



def Rmse_metric(X, Y):
    N=len(X.flatten())
    dS = (X.flatten() - Y.flatten())
    Rmse_value = np.sqrt((1/N)*dS.dot(dS))
    return Rmse_value

def canberra_metric(X, Y):
    N=len(X.flatten())
    d = np.abs(X.flatten() - Y.flatten()) / (np.abs(X.flatten()) + np.abs(Y.flatten()))
    canberra_value = np.nansum(d)/N
    return canberra_value



def KL(P,Q):
    epsilon = 0.00001

    P = np.asarray(P, dtype=np.float)
    Q = np.asarray(Q, dtype=np.float)

    P=P.reshape(-1)
    Q=Q.reshape(-1)
    
    P[P<0] = 0
    Q[Q<0] = 0
        
    P = P+epsilon
    Q = Q+epsilon

    KL_divergence = np.sum(P*np.log(P/Q))
    
    return KL_divergence


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))    


def compute_probs(data, n=100): 
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q): 
    sup_int = list(
                filter(
                    lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
                )
    )
    return sup_int

def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

# def kl_divergence(p, q):
#     return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
 
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)

def compute_kl_divergence(train_sample, test_sample, n_bins=100): 
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    
    return kl_divergence(p, q)

def compute_js_divergence(train_sample, test_sample, n_bins=100): 
    """
    Computes the JS Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)
    
    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)
    
    return js_divergence(p, q)    

