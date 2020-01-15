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
