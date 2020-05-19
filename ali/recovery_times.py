import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import sys

from ali.ali_peaks import avCurves

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

def findRecoveryPoints(pChamber : np.array, pAct : float, th : int = 10) -> np.array:
    """Compute recovery time until a specified pressure level is reached
    Input:
    pChamber : np.array
        pressure line where to search into
    pAct : float = 0
        pressure level to reach.
    Output:
        a : np.array
        recovery point indices for the whole timeline"""

    a = np.where(pChamber < pAct)[0]
    a = filterRepeatedOccurrences(a, th = th)
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


def peakRecoveryTimes(pChamber : np.array, pAct : float) -> np.array:
    """Compute recovery time until a specified pressure level is reached
    Input:
    pChamber : np.array
        pressure line where to search into
    pAct : float = 0
        pressure level to recover.
    Output:
        rT : np.array
        recovery times for the whole timeline"""

    a = np.where(pChamber < pAct)[0]
    b = filterRepeatedOccurrences(a)
    if len(b) == 0 :     # Can happen if the peak starts over pAct
        return np.nan
    else :
        return b/100

def plotRecoveryTimes(dfp : pd.DataFrame, pAct : float , ax : plt.axes = plt.gca(),
                     col : str = 'b', lb : str = '') -> list:
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
    for i in dfp.columns:
        rPt = findRecoveryPoints(dfp[i], pAct)
        if len(rPt) > 0:
            rT.append(rPt[0]/100)
    if ax == None:
        ax = plt.gca()
    ax.plot(rT, 'o', color=col, label=lb)
    ax.set_xlabel('pulse nr.')
    ax.set_ylabel('recovery time [s]')
    return rT

def shorten_dfp(dfp : pd.DataFrame, pThres : float, inTh : int = 5):
    """Crop dfp of ALI pulse up to time to recover pThres pressure
    Input:
    ---------------
    dfp : ALI pulses df
    pThres : pressure level to recover
    inTh : minimum number of index points
        for recPoints to be separated

    Returns:
    dfShort : shortened dfp
    ---------------
    """
    rP = []
    dfShort = pd.DataFrame()
    Npeaks = len(dfp.columns)
    for i in dfp:
        rPt = findRecoveryPoints(dfp[i], pAct=pThres, th=inTh)
        if rPt.any():
            rP.append(rPt[0])
            dfShort[i] = dfp[i][:rPt[0]]
    return dfShort

def hard_shorten_dfp(dfp: pd.DataFrame, tmax: float) -> pd.DataFrame:
    """Manual crop pulses to a maximum duration tmax [s]
       Useful for experiments when the pulse valve couldn't close
       and the pressure take forever to recover"""
    tstep = 0.01 # [log time in seconds]
    indMax = int(tmax / tstep)
    dfShort = pd.DataFrame()
    for i in dfp:
        dfShort[i] = dfp[i][:indMax]
    return dfShort

def plot_maxP(dfp : pd.DataFrame, ax = None, flag_p : bool = False, col = 'b', lb : str = '') -> list:
    """Compute and plot max reached pressures in a dfPeak
    Return list of maxP
    Input:
    dfp : pd.DataFrame
        ALI peaks df (from .pyk file)
    Output:
    maxP : list
        List of max. pressures reached per peak"""
    import re
    maxP = dfp.max(axis=0)
    pNr = [int(re.findall(r'(\d+)', i)[0]) for i in maxP.index]
    maxP.index = pNr
    if flag_p:
        if ax == None: ax = plt.gca()
        ax.semilogy(maxP, 'o', color = col, label=lb)
        ax.set_xlabel('pulse nr.')
        ax.set_ylabel('$p_{max}$ [mbar]')
    return maxP

def split_wet_dry(dfp : pd.DataFrame, nsigma : float = 1., DEBUG : bool = False)->tuple:
    """
        Split run by chi2 soft thresholding
        Input:
        ------
        dfp : pulse df to split
        nsigma : nr of std deviations over/below average to set chi2 threshold
            if negative reverse dry <-> wet
        Output:
        -------
        dryPulses : dfp with lower p_max (bkg)
        wetPulses : dfp with higher p_max
    """
    maxP = plot_maxP(dfp)
    x = np.arange(len(maxP))
    UmaxP = 0.3 * np.array(maxP)         # Systematic error of press gauge

    from scipy.optimize import curve_fit
    def const (x, a): return a          # fit to constant value to extract chi2

    fit, cov = curve_fit(const,x, maxP)
    assert fit[0] > np.min(maxP) and fit[0] < np.max(maxP), "av maxP out of bounds"

    chi2 = (maxP - fit[0])**2/UmaxP**2
    sd = np.std(chi2)
    avg = np.average(chi2)

    if DEBUG:
        plt.axhline(y=avg, color='r')
        plt.plot(x, chi2, '.')
        plt.fill_between(x, avg - nsigma*sd, avg + nsigma*sd, color='r', alpha=0.3)
        plt.title('Max pressure per pulse')

    wetId = np.where(chi2 < avg + nsigma * sd)[0]
    dryId = np.where(chi2 >= avg + nsigma * sd)[0]

    wetPulses = pd.concat((dfp.iloc[:,p] for p in wetId), axis=1)
    dryPulses = pd.concat((dfp.iloc[:,p] for p in dryId), axis=1)
    return wetPulses, dryPulses

def table_vnm_pulses(Ntot : int, Nwet : int,
                     C : float, M_ba : float, Vload : float = 0):
    """
    Tabulate Volume, Number of molecules and mass injected per load, pulse
    and wet pulse (assuming all solution injected only in wet pulse)
    Input:
    Ntot : total number of pulses
    Nwet : total number of wet pulses
    C : solution concentration (specify units)
        example: C = 0.5 * milimole / liter # example
    M_ba : molecular mass [specify units]
        example: M_ba = 336.24 * gram/mol # molec mass [g/mol] for Ba(ClO4)2
    Vload : volume of ALI load (default 0.3 ml)
    """
    from invisible_cities.core.system_of_units import ml, liter
    from invisible_cities.core.system_of_units import milimole, mol, gram, milligram
    NA = 6.023e23 # Avogadro constant [molec/mol]

    if Vload == 0: Vload = 0.3 * ml

    vols = (Vload / ml, Vload / Ntot / ml, Vload / Nwet / ml)
    Nmolecs = tuple([v * C / milimole * NA for v in vols])
    masses = tuple([v * C / milimole * M_ba / gram for v in vols])

    print(' \t total load \t per pulse \t per wet pulse')
    print('V [mL]\t %.2e \t %.2e \t %.2e' %(vols))
    print('N \t %.2e \t %.2e \t %.2e' %(Nmolecs))
    print('m [g] \t %.2e \t %.2e \t %.2e' %(masses))

### Area integration, leak rate and leaked mass

def LeakMass (wetP : pd.DataFrame(),
                M : float = 41.05,
                Seff : float = 0.25) :
    """Integrate area under ALI pressure curves and compute leak rate and leaked mass.
        Parameters
        ----------
        wetP : ALI pandas.Dataframe (wet pulses)
        M : solution/gas molarity. Default value for AcN
        Seff : pumping effective speed. Default value 0.25 for C0 = 0.26
        Returns
        ----------
        leak_mass : average leaked mass in gramm
        ql : average leak rates per pulse [mbar*L/s]
    """

    logtime = 0.01 # [10 ms]
    T = 25+273 # Pulse valve temperature [K]
    R = 83.144598 # Gas constant [L⋅mbar⋅K−1⋅mol−1]

    avWet, sdWet, _ = avCurves(wetP, )
    wetArea = area_raise(wetP)
    dt = np.argmax(avWet)/100

    ql = np.average(wetArea) * Seff
    leak_mass = ql * dt * M / (R*T)
    return leak_mass, ql

def area_raise(dfp : pd.DataFrame) -> list:
    """Numerical integration of pulses until array maximum (peak) for steady leak rate"""
    # For one pulse:
    # np.trapz(dfp.pulse0.iloc[:dfp.pulse0.idxmax()], x=dfp.pulse0.index[:dfp.pulse0.idxmax()]/100 )
    upAreas = [np.trapz(dfp[p].iloc[:dfp[p].idxmax()], x=dfp[p].index[:dfp[p].idxmax()]/100 ) for p in dfp]
    return upAreas

def area_Pdot(dfp: pd.DataFrame) -> np.array:
    """Compute integrated area under time derivative chamber pressure for dynamic leak rate"""
    areaPdot = []
    tstep = 100

    for p in dfp:
        tmax = dfp[p].idxmax()+1
        pRise = dfp[p].iloc[:tmax]
        pdot = (pRise[1:].values - pRise[:-1].values)/tstep
        areaPdot.append(np.trapz(pdot, x=dfp.index[:tmax-1]/100) )

    return areaPdot

def avLeakN (wetP : pd.DataFrame(),
            Seff : float = 0.25,
            Vcham: float = 4.7) :
    """Integrate area under ALI pressure curves and
        compute average leak rate and number of leaked molecules per pulse.
        Parameters
        ----------
        wetP : ALI pandas.Dataframe (wet pulses)
        Seff : pumping effective speed. Default value 0.25 for C0 = 0.26
        Vcham : chamber volume. Default value 4.7 L
        Returns
        ----------
        (leakN, uleakN) : tuple average leaked molecules and uncertainty
        (qL, uqL) : tuple average leak rates per pulse [mbar*L/s] (steady and dynamic) and error
    """

    logtime = 0.01 # [10 ms]
    T = 25+273 # Pulse valve temperature [K]
    R = 83.144598 # Gas constant [L⋅mbar⋅K−1⋅mol−1]
    NA = 6.023e23 # Avogadro constant [molec/mol]

    avWet, sdWet, _ = avCurves(wetP)
    wetArea = area_raise(wetP)
    areaPdot = area_Pdot(wetP)
    dt = np.argmax(avWet)/100

    qLsted = np.average(wetArea) * Seff
    uqLsted = np.std(wetArea)/np.sqrt(len(wetArea)) * Seff

    qLdyn = np.average(areaPdot) * Vcham
    uqLdyn = np.std(areaPdot)/np.sqrt(len(areaPdot)) * Vcham

    qL = qLsted + qLdyn
    uqL = np.sqrt(uqLsted**2 + uqLdyn**2)

    leakN = qL * dt * NA / (R*T)
    uleakN = uqL * leakN/qL
    return (leakN, uleakN), (qL, uqL)

def totalLeakN (wetP : pd.DataFrame(),
            Seff : float = 0.25,
            Vcham: float = 4.7) :
    """Integrate area under ALI pressure curves and compute total leak rate and number of leaked molecules.
        Parameters
        ----------
        wetP : ALI pandas.Dataframe (wet pulses)
        Seff : pumping effective speed. Default value 0.25 for C0 = 0.26
        Vcham : chamber volume. Default value 4.7 L
        Returns
        ----------
        (leakN, uleakN) : tuple average leaked molecules and uncertainty
        (qL, uqL) : tuple average leak rates per pulse [mbar*L/s] (steady and dynamic) and error
    """

    logtime = 0.01 # [10 ms]
    T = 25+273 # Pulse valve temperature [K]
    R = 83.144598 # Gas constant [L⋅mbar⋅K−1⋅mol−1]
    NA = 6.023e23 # Avogadro constant [molec/mol]

    wetArea = np.array(area_raise(wetP) )
    areaPdot = np.array( area_Pdot(wetP) )
    dt = wetP.idxmax().values/100
    udt = logtime   # Relative error up to 10%

    qLsted = wetArea * Seff
    sqLsted = udt* wetArea * Seff

    qLdyn = areaPdot * Vcham
    uqLdyn = areaPdot * udt * Vcham

    qL = qLsted + qLdyn
    sqL = np.sqrt(uqLsted**2 + uqLdyn**2)

    leakPulse = qL * dt * NA / (R*T)
    sleakPulse = leakPulse * np.sqrt( (sqL/qL)**2 + (udt/dt)**2 )

    leakN = np.sum(leakPulse)
    sleakN = np.sqrt(np.sum(sleakPulse**2))
    return (leakN, sleakN), (qL, sqL), (leakPulse, sleakPulse)

def compute_soluteSolvent_ratio(conc: float, rho_mol : float, rho_solv : float, M_mol : float, M_solv : float):
    """Compute n_solute/n_solvent ratio
        Parameters:
        ----
        conc: solution concentration
        rho_mol: solute (molecule) mass density
        rho_solv: solvent mass density
        M_mol: solute molar mass
        M_solv: solvent molar mass
        Use coherent units"""
    inverse = M_solv/rho_solv*(1/conc - rho_mol/M_mol)
    return 1/inverse

rho_acn = 1.67
M_acn = 41.05
rho_baclo = 3200 # g/L
M_baclo = 336.3
rho_fbi = 21.6
M_fbi = 529
conc = 1e-6 #Molar

def fit_pump_profile(dfp : pd.DataFrame, xdown : list, pdown : list):
    """Fit custom model to average pumping profile of ALI curve
    Model is the sum of two exponential with different decay and a negative power of t
    Input:
    --------
    dfp : ALI curve to extract the average profile
    xdown : array of descend time to perform fit on
    pdown : array of descend pressure to perform fit on"""

    plotAverageProfile(dfp, lb='Average wet profile')

    from scipy.optimize import curve_fit
    def exp2(x, p0, tau, p1, n, p2, beta):
        return p0*np.exp(-tau*(x)) + p1*(x)**n + p2*np.exp(-beta*x)

    fitDw, covDw = curve_fit(exp2, xdown, pdown, p0=[pdown[0], 30, pdown[-1], 0.1, 5e-3, 0.5])
    plt.semilogy(xdown, exp2(xdown, *fitDw), '-r', label='Fit model')
    plt.legend()
    return fitDw #, np.sqrt(np.diag(covDw))
