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


from ali.ali_sw import load_raw_ali_df

def findTroughsFast(dfRaw) -> np.array:
    """Faster trough-finding routine based on dataframe entry 'valve' = on/off"""

    tr0 = np.where(dfRaw['valve'] == 1)[0]
    tr1 = tr0[np.where(tr0[1:] - tr0[:-1] > 1)[0]] - 1 # Several consecutive entries may have 'valve' on, pick only the first (darum -1) of each set
    troughs = np.append(tr1, tr0[len(tr0)-1])  # The last peak must be included manually (can never fulfill the previous condition)

    return troughs

def extractDfPeaks(dfRaw : pd.DataFrame, troughs : np.array, peakLength : int = 0, full : bool = True) -> pd.DataFrame :
    """Transform dfpRaw into matrix and dfPeaks
    Parameters:
    peakLengthCut : int
        Index slice size per peak. If no value is given, computed as minimum
        distance between peaks."""

    if full:  #In this case import whole peak by looping
        m0 = []
        for i, t in enumerate(troughs):
            if i < len(troughs) - 1:
                m0.append(dfRaw.p_chamber[t : troughs[i+1]].values)

        m0.append(dfRaw.p_chamber[t :].values)
        dfpeak = pd.DataFrame(list(map(np.ravel, m0))).transpose()
        dfpeak.columns = ['pulse'+str(i) for i in range(len(troughs))]

    else:
        if peakLength == 0:
            peakLength = int(np.min(troughs[1:] - troughs[:-1]))

        lenP = len(dfRaw.p_chamber)
        troughs = troughs[lenP - troughs > peakLength] # exclude peaks in the end with too short length

        npeaks = troughs.shape[0]
        indices_offset = np.repeat(np.arange(peakLength), npeaks).reshape(peakLength, npeaks)

        dfpeak = pd.DataFrame(dfRaw.p_chamber.values[indices_offset + troughs])
        dfpeak.columns = ['pulse'+str(i) for i in range(len(troughs))]

    return dfpeak

def process_dfRaw_peaks(path : str, peakLength : int = 0, full : bool = True) -> pd.DataFrame:
    """Method to process ALI raw df into peak df
    Parameters:
    peakLengthCut : int
        Index slice size per peak. If no value is given, computed as minimum
        distance between peaks."""

    dfRaw = load_raw_ali_df(path)
    troughs = findTroughsFast(dfRaw)
    dfpeak = extractDfPeaks(dfRaw, troughs, peakLength = peakLength, full = full)

    return dfpeak

def avCurves(dfp : pd.DataFrame):
    """Compute average curve profile, uncertainty area and residue dataframe
    Input:
    -----
    dfp : pd.DataFrame
        Processed peaks from ALI dfRaw"""
    av_p = np.array(dfp.mean(axis=1))
    sd_p = np.array(dfp.std(axis=1))

    dfResidue = av_p.reshape(len(av_p),1) - dfp
    return av_p, sd_p, dfResidue

def thumbnailPulses (dfp : pd.DataFrame, figw : int = 20, figh : int = 20):
    """Plot a thumbnail mosaic of the pulses in dfp
    Ncol and Nrows are automatically computed from the sqrt of Npeaks"""

    Npeaks = len(dfp.columns)
    Ncol = int(np.floor(np.sqrt(Npeaks)))
    Nrow = Ncol + 1
    Nrow, Ncol, Npeaks
    fig, ax = plt.subplots(Nrow, Ncol, figsize=(figw, figh))
    x = dfp.index.values

    for i, r in enumerate(dfp.columns):
        j = i//(Nrow-1)
        k = i%Ncol

        y = dfp[r].values

        ax[j,k].plot(x, y, '-')
        ax[j,k].set_yscale('log')
        ax[j,k].set_xticks([])
        ax[j,k].set_yticks([])
    return ax

def cosmetics_aliplot(ax = None):
    if ax == None: ax = plt.gca()
    ax.set_yscale('log')
    ax.set_ylabel('$p_{chamber}$ [mbar]')
    ax.set_xlabel('$t$ [s]')
    ax.legend()

def overplot_every_N_pulses(dfp : pd.DataFrame, every : int = 20, ax = None):
    Npeaks = len(dfp.columns)
    x = dfp.index.values/100

    if ax == None: ax = plt.gca()
    for i in dfp.iloc[:, ::every]:
        ax.plot(x, dfp[i], label=i)
    cosmetics_aliplot()
    return ax

def plotResiduesDistribution(dfp : pd.DataFrame, nsigma : int = 2) -> list:
    """Plot and fit to gaussian the total residuals (integrated over whole time)
    Return ID of peaks whose residues lay out of the nsigma CI"""
    import matplotlib.pyplot as plt
    _,_, dfResidue = avCurves(dfp)
    res_peak = dfResidue.sum(axis=0)

    plt.hist(res_peak, bins=10, zorder=-1)

    n, bins = np.histogram(res_peak, bins=10)
    cbins = 0.5*(bins[1:]+bins[:-1])

    from scipy.stats import norm
    (mu, sigma) = norm.fit(res_peak)
    mu, sigma
    xpl = np.linspace(min(cbins), max(cbins), 100)
    ypl = norm.pdf(xpl, mu, sigma)

    xci = np.linspace(mu- nsigma * sigma, mu + nsigma * sigma, 100)
    yci = norm.pdf(xci, mu, sigma)
#         ypl *= n.sum()
#         yci *= n.sum()

    plt.plot(xpl, ypl, 'r', label='Gaussian fit \nmean = %.2e \nsigma = %.2e' %(mu, sigma))
    plt.fill_between(xci, 0, yci, alpha=0.9, color='g', zorder = 1, label='2$\sigma$ CI')
    plt.xlabel('Peak total residuals')
    plt.legend()

    peaksID = np.where(dfResidue.sum(axis=0) < mu - nsigma* sigma)[0]

    return ['peak'+str(i) for i in peaksID]

def plotAverageProfile(dfp : pd.DataFrame, flag_p: bool = False, nsigma : int = 1,
 ax = None, lb : str = '', lb2 : str = ''):
    """Plot average curve profile and the peaks with residuals larger than the specified Confidence Level
    Parameters:
    flag_p: bool
        Flag to include irregular peaks. Default: False
    nsigma: int
        Level of confidence for peaks total residuals. Default: False"""
    import matplotlib.pyplot as plt
    sc = 100 # Timescale: seconds


    av_p, sd_p, _ = avCurves(dfp)

    if ax == None:
        ax = plt.gca()

    if flag_p:
        peaksID = plotResiduesDistribution(dfp, nsigma);
        ax.cla()

        for p in peaksID:
            ax.plot(dfp.index/sc, dfp[p].values, label = p)

    color = ax.plot(dfp.index/sc, av_p, label = lb)[0].get_color()
    ax.fill_between(dfp.index/sc, av_p-nsigma*sd_p, av_p+nsigma*sd_p, color=color, alpha=0.3, label=lb2)

    cosmetics_aliplot()
    return ax

def polishDfPeak(dfp, nsigma : int = 2) -> pd.DataFrame:
    """Drop peaks with total residual lying outside of the specified confidence interval
    Returns dfpeak with dropped columns"""

    peaksID = plotResiduesDistribution(nsigma);
    plt.clf();

    dfp.drop(['peak'+str(i) for i in peaksID], axis=1, inplace=True)
    return dfp

def save_processed_peaks(path: str, dfp : pd.DataFrame):
    """Save dfp with .pyk extension to subdirectory 'processed_peaks' of current directory,
    create it if it does not exist"""

    path_array = os.path.split(path)
    path_pyk = path_array[0] + '/processed_peaks/' + path_array[1] + '.pyk'
    try :
        print('Saving processed peaks df to ', path_pyk)
        dfp.to_csv(path_pyk)
    except FileNotFoundError:
        print('Lets try soming else')
        print('processed_peaks subdirectory did not exist yet, creating it...')
        os.mkdir(path_array[0] + '/processed_peaks')

        print('Saving processed peaks df to ', path_pyk)
        dfp.to_csv(path_pyk)

@dataclass
class AliPeaks:
    "ALI dataclass, peaks df and metadata"
    dfp : pd.DataFrame
    p_gas : str
    t_on : str
    date : str
    other_meta : str = None

def import_peaks(path : str) -> list:
    """Preliminar import peaks dfp and metadata
    Should go into AliPeaks class
    Returns: dfp, p_gas, t_on, date"""
    dfp = pd.read_csv(path, index_col = 0)
    filename = os.path.split(path)[1]
    p_gas  = re.search('\d+mbar', filename).group(0)
    t_on = re.search('\d+ms', filename).group(0)
    da = re.search('/\d+_', path).group(0).replace('/', '').replace('_', '')
    date = re.sub('(\d{2})(\d{2})(\d{4})', r"\1.\2.\3", da, flags=re.DOTALL)

    other_meta = filename.replace(p_gas, '').replace(t_on, '').replace(date, '')

    return AliPeaks(dfp, p_gas, t_on, date, other_meta)
