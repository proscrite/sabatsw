import pandas as pd
import numpy as np

import glob
import os
import sys

def filterRepeatedOccurrences(a : np.array, th: int = 10) -> np.array:
    """Select first occurrence in an array of indices
    by computing backwards the distance between them and filtering out the immediate neighbours
    Input:
    a : np.array
        array of indices with the position of the neighboring occurrences
    th : int = 10
        threshold of minimum distance between desired occurrences"""

    b = np.where(a[-1:0:-1] - a[-2::-1] > 10)[0]       # Since we iterated backwards, the resulting b array is also in the reverse order, so we flip it
    return np.flip(a[len(a)-1 - b])

def findRecoveryPoints(pChamber : np.array, pAct : float) -> np.array:
    """Compute recovery time until a specified pressure level is reached
    Input:
    pChamber : np.array
        pressure line where to search into
    pAct : float = 0
        pressure level to reach."""

    a = np.where(pChamber < pAct)[0]
    a = filterRepeatedOccurrences(a)
    return a

def computeRecoveryTimes(recPoints : np.array, scale : str = 's') -> np.array:
    """From array of recovery time point indices, compute recovery times as distance in time between these"""
    if scale == 'ms':
        sc = 0.1
    elif scale == 'min':
        sc = 6000
    else: sc = 100

    recTimes = recPoints[1:] - recPoints[:-1]
    return recTimes/sc

def computePAct(df : pd.DataFrame, troughs : np.array, flag_df : bool = False) -> float:
    """Estimate actuation pressure from the values of the troughs positions
    If flag_df is activated, the value in p_act will be taken (useful if ALI enters in OFF time)"""

    if flag_df: p_act = df.p_act[0]
    else:
        p_act = np.average(df.p_chamber[troughs]) + np.std(df.p_chamber[troughs])
    return p_act


#### To use exclusively with dfp (peaks df) NOT with dfRaw  ####

def plotRecoveryTimes(dfp : pd.DataFrame, pAct : float) -> list:
    """Compute and plot recovery times from a dfpPeak and specified pAct
    Return recovery time
    Input:
    dfp : pd.DataFrame
        ALI peaks df (from .pyk file)
    pAct : float
        pressure threshold for recovery time
    """
    rT = []
    Npeaks = len(dfp.columns)
    for i in range(Npeaks):
        rPt = findRecoveryPoints(dfp['peak'+str(i)], pAct)
        if len(rPt) > 0:
            rT.append(rPt[0]/100)
    plt.plot(rT, 'o')
    plt.xlabel('peak nr.')
    plt.ylabel('recovery time [s]')
    return rT

def plot_maxP(df : pd.DataFrame) -> list:
    """Compute and plot max reached pressures in a dfPeak
    Return list of maxP
    Input:
    dfp : pd.DataFrame
        ALI peaks df (from .pyk file)
    Output:
    maxP : list
        List of max. pressures reached per peak"""

    maxP = []
    for i in range(Npeaks):
        maxP.append(np.max(df['peak'+str(i)]))
    plt.semilogy(maxP, 'o')

    plt.xlabel('peak nr.')
    plt.ylabel('max p_chamber [mbar]')
    return maxP

def depletionFromMaxP(dfp) -> int:
    """Find liquid exhaustion moment by fitting a constant to the maxP
    scatter and choosing the maximum value of chi2. Plot fit and Chi2"""

    maxP = plot_maxP(df)
    x = np.arange(len(maxP))
    UmaxP = 0.3 * np.array(maxP)
    from scipy.optimize import curve_fit
    def func (x, a): return a

    fit, cov = curve_fit(func,x, maxP)#, sigma=UmaxP, absolute_sigma=True)
    sd = np.std(maxP)
    # sd = cov[0]
    chi2 = (maxP - fit[0])**2/UmaxP**2

    plt.axhline(y=fit[0], color='r')
    plt.plot(x, chi2, '.')
    # plt.fill_between(x, fit[0]-cov[0], fit[0]+cov[0], color='g', alpha=0.5)
    plt.fill_between(x, fit[0]-sd, fit[0]+sd, color='r', alpha=0.3)
    plt.title('Max pressure per pulse')
    return np.argmax(chi2)
