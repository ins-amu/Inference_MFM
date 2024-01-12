#!/usr/bin/env python3
"""
@author: meysamhashemi  INS Marseille

"""
import os
import sys
import numpy as np
import scipy as scp

from scipy import signal
from scipy.signal import  find_peaks, peak_widths
from scipy import stats as spstats
from scipy.stats import moment
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import mode


def calculate_summary_statistics(v, dt, ts, t_on, t_off):
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
   

    peaks, _ = find_peaks(x_act, height=.2, distance=1000)
    spikes_num=peaks.shape[0]
        
    if spikes_num > 0:
        spikes_onset=t_on+dt*peaks[0]
    else:            
        spikes_onset=0.
                                
                        
    sum_stats_vec=np.concatenate((sum_stats_vec, 
                np.array(spikes_num).reshape(-1), 
                np.array(spikes_onset).reshape(-1)  
               ))

    sum_stats_vec = sum_stats_vec[0:n_summary]        

    return sum_stats_vec



########################################################################################################################

def oscil_class(v):
    nlen=int(v.shape[0])
    oscillation = ''
    # Check for oscillatory behavior
    oscillatory = False
    for i in range(2, nlen-1):
            if (v[i] - v[i-1]) * (v[i+1] - v[i]) < 0:                
                oscillatory = True
                break
                
    if not oscillatory:
            oscillation = 'Sink'
    else:
            # Check for damping behavior
            damping = False
            for i in range(2, nlen):
                if abs(v[i] - v[i-1]) < abs(v[i+1] - v[i]):
                    damping = True
                    break
            if damping:
                oscillation = 'Damped'
            else:
                oscillation = 'Exploded'
    return  oscillation       

########################################################################################################################


def calculate_summary_statistics_dynamicalclass(v, dt, ts, t_on, t_off):
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

    x_act=x[(ts > t_on) & (ts < t_off)]
    x_pre=x[(ts < t_on)]
    x_post=x[(ts > t_off)]

    sum_stats_vec=[]
    
    W_pre=x_pre[20000:]
    W_act=x_act[20000:]
    W_post=x_post[20000:]
    
    S1_act=x_act[0:5000]
    S2_act=x_act[-5000:-100]

    peaks_W_pre, _ = find_peaks(W_pre, rel_height=0.01, width=10)
    peaks_W_act, _ = find_peaks(W_act, rel_height=0.01, width=10)
    peaks_W_post, _ = find_peaks(W_post, rel_height=0.01, width=10)

    spikes_num_W_pre=int(peaks_W_pre.shape[0])
    spikes_num_W_act=int(peaks_W_act.shape[0])
    spikes_num_W_post=int(peaks_W_post.shape[0])
    
    peaks_S1_act, _ = find_peaks(S1_act, rel_height=0.01, width=10)
    peaks_S2_act, _ = find_peaks(S2_act, height=0.2)

    spikes_num_S1_act=int(peaks_S1_act.shape[0])
    spikes_num_S2_act=int(peaks_S2_act.shape[0])

    if    (spikes_num_W_pre==0  and spikes_num_W_act==0 and  spikes_num_W_post==0  and oscil_class(W_pre)=='Sink'   and  oscil_class(W_act)=='Sink'    and  oscil_class(W_post)=='Sink' and  np.round(np.mean(x_pre[-100:]), 2)==np.round(np.mean(x_post[-100:]), 2)):  
               spike_class=0.   
            
    elif  (spikes_num_W_pre>0   and spikes_num_W_act>0  and oscil_class(W_pre)=='Damped' and  oscil_class(W_act)=='Damped'):
               spike_class=1.
           
    elif   (spikes_num_W_pre==0  and spikes_num_W_act>0  and spikes_num_W_post>0  and oscil_class(W_pre)=='Sink'   and  oscil_class(W_act)=='Damped'  and  oscil_class(W_post)=='Damped'):  
               spike_class=2.  

    elif   (spikes_num_W_pre==0  and spikes_num_W_act>0  and spikes_num_W_post==0  and oscil_class(W_pre)=='Sink'   and  oscil_class(W_act)=='Damped'  and  oscil_class(W_post)=='Sink'):  
               spike_class=3.  
    else:
               spike_class=4.
        
    sum_stats_vec=np.concatenate((sum_stats_vec, np.array(spike_class).reshape(-1), ))
    

    sum_stats_vec = sum_stats_vec[0:n_summary]        


    return sum_stats_vec

########################################################################################################################

