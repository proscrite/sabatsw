import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import sys
import re

import datetime
import warnings
from dataclasses import dataclass

sys.path.append('~/sabat/sabatsw')

def import_qmb_data(data : str) -> pd.DataFrame:

    df = pd.read_csv(data, sep='|', skiprows=2, engine='python', names = ['timestamp', 'livetime', 'statusline', 'run', 'logstate', 'other'])
    other = df.other.str.split(',', expand=True)
    datet = df.timestamp.str.split(',', expand=True)[0]
    other.dropna(inplace=True)
    df['timestamp'] = datet.apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S'))
    df['rate'] = (other[1].map(str) + '.' + other[2].map(str)).apply(lambda x: float(x))
    df['film_thickness'] = (other[3].map(str) + '.' + other[4].map(str)).apply(lambda x: float(x))
    df['substrate_thickness'] = (other[5].map(str) + '.' + other[6].map(str)).apply(lambda x: float(x))
    df['frequency'] = (other[7].map(str) + '.' + other[8].map(str)).apply(lambda x: float(x))
    df.drop('other', axis=1, inplace=True)
    return df

def compute_Nba(thick : float) -> float:
    """For a pellet with 1.3 cm diameter compute number of barium molecules from thickness [in cm]
    Input: film thickness from QMB [cm]"""

    d_pel = 1.3 # pellet diameter [cm]
    d_qc = 0.8 # quartz crystal diameter [cm]
    r = d_pel / d_qc
    S_pel = np.pi*(d_pel/2)**2  # pellet area [cm^2]

    V_ba = thick * S_pel
    M_ba = 336.24 # molec mass [g/mol]
    p_ba = 3.2  # molecular density [g/cm^3]
    N_A = 6.023e23 # Avogadro constant [molec/mol]
    m_ba = V_ba * p_ba  # Deposited mass
    N_ba = m_ba / M_ba * N_A   # Number of molecules deposited
    print('Total number of Barium perchlorate molecules deposited = %.2e' %N_ba)
    return N_ba

def extractQMBPeaks(dfRaw : pd.DataFrame, troughs : np.array, peakLength : int = 0, full : bool = True) -> pd.DataFrame :
    """Transform dfpRaw into matrix and dfPeaks
    Parameters:
    peakLengthCut : int
        Index slice size per peak. If no value is given, computed as minimum
        distance between peaks."""

    if full:  #In this case import whole peak by looping
        m0 = []
        for i, t in enumerate(troughs):
            if i < len(troughs) - 1:
                m0.append(dfRaw.frequency[t : troughs[i+1]].values)

        m0.append(dfRaw.frequency[t :].values)
        dfpeak = pd.DataFrame(list(map(np.ravel, m0))).transpose()
        dfpeak.columns = ['pulse'+str(i) for i in range(len(troughs))]

    else:
        if peakLength == 0:
            peakLength = int(np.min(troughs[1:] - troughs[:-1]))

        lenP = len(dfRaw.frequency)
        troughs = troughs[lenP - troughs > peakLength] # exclude peaks in the end with too short length

        npeaks = troughs.shape[0]
        indices_offset = np.repeat(np.arange(peakLength), npeaks).reshape(peakLength, npeaks)

        dfpeak = pd.DataFrame(dfRaw.frequency.values[indices_offset + troughs])
        dfpeak.columns = ['pulse'+str(i) for i in range(len(troughs))]

    return dfpeak

def overplot_QMB_pulses(dfq : pd.DataFrame, every : int = 20, ax = None):
    Npeaks = len(dfq.columns)
    x = dfq.index.values

    if ax == None: ax = plt.gca()
    for i in dfq.iloc[:, ::every]:
        ax.plot(x, dfq[i], label=i)
    ax.set_ylabel('f [Hz]')
    ax.set_xlabel('t [s]')
    ax.legend()
    return ax

def qmb_troughs_asALI(dfq: pd.DataFrame, dfp: pd.DataFrame, minAli: np.array, offset: int = 3) -> np.array:
    """Find closest points to ALI troughs (minAli) in  synchronized dfq
        Offset of -3 s by default"""
    minQmb = []
    for tr in dfp.datetime[minAli].values:
        minQmb.append(np.argmin(abs(dfq.timestamp.values - tr)) - offset)

    return minQmb

def coating_mass(dfq : pd.DataFrame, upCut: int = 10):
    """Compute frequency shift and coating mass in QMB deposition"""

    Fq = 6045000 # Blank frequency [Hz]
    Nat = 166100 # Frequency constant [Hz*cm]
    pq = 2.649 # Single crystal quartz density [g/cm^3]
    pf = 3.2 # Film density for Ba(ClO_4)_2 [g/cm^3]

    K = Nat * pq / Fq**2   # Empirical constant [g/(cm^2 * Hz)]
    d_qc = 0.8 # quartz crystal diameter [cm]
    A_q = np.pi*(d_qc/2)**2  # quartz crystal area [cm^2]
    mq = K * Fq * A_q   # Quartz crystal mass [g] (reference)

    deltaF = []
    mL = []
    for pulse in dfq:
        try:
            tend = dfq[pulse].iloc[:upCut].idxmin()
            tstart = dfq[pulse].iloc[:tend].idxmax()
            freqstart = dfq[pulse].iloc[tstart]
            freqend = dfq[pulse].iloc[tend]
            dF = freqstart - freqend
            deltaF.append(dF)

            mL.append(mq * dF/freqstart)
            continue
        except ValueError:
            print(pulse + ' is too short with upCut ' + str(upCut))
            continue

    return mL, deltaF

def plot_ali_qmb_pulses(dfp: pd.DataFrame, dfq: pd.DataFrame, ax = None, tfraction = None):
    if ax == None: ax = plt.gca()
    ax.plot(dfq.timestamp, dfq.frequency)
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_ylabel('f [Hz]')

    if tfraction is not None:
        ax.set_xlim(dfq.timestamp[0], tfraction)

    ax2 = plt.twinx(ax)
    ax2.plot(dfp.datetime, dfp.p_chamber, '-r')
    ax2.set_ylabel('p$_{chamb}$ [mbar]')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.autofmt_xdate()
