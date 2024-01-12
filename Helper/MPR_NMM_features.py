#!/usr/bin/env python3
"""
@author: meysamhashemi  INS Marseille

"""
import os
import sys
import numpy as np
import scipy as scp

from scipy import signal
from scipy.signal import  find_peaks, peak_widths, savgol_filter
from scipy import stats as spstats
from scipy.stats import moment
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import mode
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from sklearn.preprocessing import minmax_scale 


def calculate_summary_statistics(v, dt, ts, t_on, t_off, features):
    """Calculate summary statistics

    Parameters
    ----------
    x : output of the simulator

    Returns
    -------
    np.array, summary statistics
    """
    
    n_summary = 100

    nt=ts.shape[0]
    x=np.zeros(nt)
    x= np.array(v)

    x_rest=x[ts < t_on]
    x_act=x[(ts > t_on) & (ts < t_off)]
    x_post= x[(ts > t_off)]

    sum_stats_vec = np.concatenate((np.array([np.mean(x_rest)]),
                                    np.array([np.std(x_rest)]),
                                    np.array([np.median(x_rest)]),
                                    np.array([skew(x_rest)]),
                                    np.array([kurtosis(x_rest)]),
                                    np.array([np.mean(x_act)]),
                                    np.array([np.std(x_act)]),
                                    np.array([np.median(x_act)]),
                                    np.array([skew(x_act)]),
                                    np.array([kurtosis(x_act)]),
                                    np.array([np.mean(x_post)]),
                                    np.array([np.std(x_post)]),
                                    np.array([np.median(x_post)]),
                                    np.array([skew(x_post)]),
                                    np.array([kurtosis(x_post)]),

                                   ))
   
    spikes_num=[]
    spikes_on=[]
    bistability=[]

    x_th=-1
    ind = np.where(x < x_th)
    x[ind] = x_th

    ind = np.where(np.diff(x) < 0)

    spike_times = np.arange(0, ts.shape[0], dt)[ind]
    spike_times_stim = spike_times[(spike_times > t_on) & (spike_times < t_off)]

    for item in features:
            if item is 'spikes':
                        if spike_times_stim.shape[0] > 0:
                                    spike_times_stim = spike_times_stim[np.append(1, np.diff(spike_times_stim)) > .5]
                                    spikes_on.append(spike_times_stim[0])
                        else:            
                                    spikes_on.append(0.)
                                
                        spikes_num.append(spike_times_stim.shape[0])  
                        
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                                        np.array(spikes_num),
                                                        np.array(spikes_on),
                                                       ))

       
            if item is 'bistability':
                        if spike_times_stim.shape[0] > 0:
                                    bistability.append(1.)
                        else:
                                    bistability.append(0.)
                                
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                                        np.array(bistability),
                                                       ))
            
    sum_stats_vec = sum_stats_vec[0:n_summary]        

    return sum_stats_vec