import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import peakutils
from copy import deepcopy
from scipy.optimize import curve_fit

from dataclasses import dataclass
from xps.xps_import import XPS_experiment
from xps.xps_analysis import plot_region, cosmetics_plot

def find_and_plot_peaks(df : pd.DataFrame, thres : float = 0.5, min_d : int = 10, col : str = 'r'):
    leny = len(df.index)
    peaks =  peakutils.indexes(df.counts.values, thres=thres, min_dist=min_d)
    x_peaks = leny - df.index[peaks]
    y_peaks = df.counts.values[peaks]
    plt.plot(x_peaks, y_peaks, col+'o', label='Peaks at thres = %.1f' %thres)

    return peaks

def scale_and_plot_spectra(xp : XPS_experiment, xpRef : XPS_experiment, region : str = 'overview_', bl_deg : int = 5, lb : tuple = None) -> float:
    """Plot two spectra and compute average count ratio between main peaks for scaling
        Input:
        -----------------
        xp: XPS_experiment
            Experiment containing the spectrum region to scale UP
        xpRef: XPS_experiment
            Reference spectrum to compare to
        lb : tuple
            Labels for legend
        Output:
        ------------------
        y_sc : array
            scaled counts
        normAv : float
            Scale factor computed as the average ratio between peak heights. Should be > 1,
            otherwise the reference spectrum has weaker intensity than the one intended to scale up
        indmax : int
            Index position of the highest peak"""
    from peakutils import indexes, baseline
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    if lb == None: lb = (xp.name, xpRef.name)
    df, dfRef = xp.dfx[region].dropna(), xpRef.dfx[region].dropna()

    ax[0].plot(df.energy, df.counts, '-b', label=lb[0])
    ax[0].plot(dfRef.energy, dfRef.counts, '-r', label=lb[1] + ' (ref.)')

    indmax = indexes(dfRef.counts.values, thres=0.99)[0] # Get only highest peak
    indmin = np.argmin(dfRef.counts[indmax : indmax + 20]) # Get absolute minimum in near neighbourhood
    ax[0].axhline(dfRef.counts[indmax], color='k', ls = '--')
    ax[0].axhline(dfRef.counts[indmin], color='k', ls = '--')
    ax[0].axvline(dfRef.energy[indmax], color='k', ls = '--')

    bl = baseline(df.counts, deg=bl_deg)
    blr = baseline(dfRef.counts, deg=bl_deg)
    ax[0].plot(df.energy, bl, '--b', label='Baseline of  ' + lb[0])
    ax[0].plot(dfRef.energy, blr, '--r', label='Baseline of ' + lb[1])

    cosmetics_plot(ax = ax[0])
    ax[0].set_title('Baseline and peak')

    # Compute normalization factor
    norm  = ( dfRef.counts[indmax] - dfRef.counts[indmin] ) / ( df.counts[indmax] - df.counts[indmin] )

    y_scale = (df.counts - bl) * norm

    ax[1].plot(df.energy, y_scale, '-b', label=lb[0])
    ax[1].plot(dfRef.energy, (dfRef.counts - blr) , '-r', label=lb[1]+ ' (ref.)')
    cosmetics_plot(ax = ax[1])
    ax[1].set_title('Scaling result')
    return y_scale, norm, indmax

def scale_dfx(xp : XPS_experiment, scale_factor : float, inplace : bool = False):
    """Rescale xp.dfx for comparison with other experiment and subtract baseline
    Returns whole XPS_experiment"""
    from peakutils import baseline

    names = list(xp.dfx.columns.levels[0])
    dfnew = pd.DataFrame()

    frames = []
    for n in names:
        bl = baseline(xp.dfx[n].dropna().counts)
        ybl = xp.dfx[n].dropna().counts - bl
        ysc = ybl.apply(lambda c : c * scale_factor)
        frames.append( pd.DataFrame([xp.dfx[n].energy, ysc]).T )
    dfnew = pd.concat(frames, axis=1)

    mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])])
    mi.to_frame()
    dfnew.columns = mi

    if inplace:
        xp.dfx = dfnew
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx = dfnew
    return xpNew

def normalise_dfx(xp : XPS_experiment, inplace : bool = False):
    """Normalise spectrum counts to maximum peak at index position indmax"""
    from peakutils import indexes
    names = list(xp.dfx.columns.levels[0])
    dfnew = pd.DataFrame()

    frames = []
    for n in names:
        y =  xp.dfx[n].dropna().counts
        ynorm = y.apply(lambda c : c / np.max(y))

        frames.append( pd.DataFrame([xp.dfx[n].energy, ynorm]).T )
    dfnew = pd.concat(frames, axis=1)

    mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])])
    mi.to_frame()
    dfnew.columns = mi

    if inplace:
        xp.dfx = dfnew
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx = dfnew
    return xpNew

def find_integration_limits(x, y, flag_plot = False, region : str = None, ax = None):
    """Utility to locate limits for shirley bg subtraction"""
    # Locate the biggest peak.
    maxidx = abs(y - np.max(y)).argmin()

    # It's possible that maxidx will be 0 or -1. If that is the case,
    # we can't use this algorithm, we return a zero background.
    if maxidx == 0 or maxidx >= len(y) - 1:
        print ("specs.shirley_calculate: Boundaries too high for algorithm: returning a zero background.")

    # Locate the minima either side of maxidx.
    lmidx = abs(y[0:maxidx] - np.min(y[0:maxidx])).argmin()
    rmidx = abs(y[maxidx:] - np.min(y[maxidx:])).argmin() + maxidx

    if flag_plot:
        if ax == None: ax = plt.gca()
        ybase = ax.get_ylim()[0]
        if ybase < 0: ybase = 0
        ind = [maxidx, lmidx, rmidx]
        for i in ind:
            ax.vlines(x = x[i], ymin=ybase, ymax=y[i], linestyles='--', color='k')
            ax.text(s= '%.2f'%x[i], x = x[i], y = y[i])

    return lmidx, rmidx

def shirley_loop(x, y,
                 lmidx : int = None,
                 rmidx : int = None,
                 maxit : int = 10, tol : float = 1e-5,
                 DEBUG : bool = False):
    """Main loop for shirley background fitting"""
    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above.

#     x, y, is_reversed = check_arrays(x, y)

    if (lmidx == None) or (rmidx == None):
        lmidx, rmidx = find_integration_limits(x, y, flag_plot=False)
    xl, yl = x[lmidx], y[lmidx]
    xr, yr = x[rmidx], y[rmidx]

    B = np.zeros(x.shape)
    B[:lmidx] = yl - yr
    Bnew = B.copy()
    it = 0
    while it < maxit:
        if DEBUG:
            print ("Shirley iteration: %i" %it)

        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = np.trapz( + B[lmidx:rmidx - 1] + yr - y[lmidx:rmidx - 1] , x=x[lmidx:rmidx - 1])
        k = (yl - yr) / ksum

        # Calculate new B
        ysum = 0
        for i in range(lmidx, rmidx):
            ysum = np.trapz( B[i:rmidx - 1] + yr - y[i:rmidx - 1] , x=x[i:rmidx - 1])
            Bnew[i] = k * ysum

        # If Bnew is close to B, exit.
        if np.linalg.norm(Bnew-B) < tol:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        it += 1

    assert it < maxit, "shirley_loop: Max iterations exceeded before convergence."
    if it >= maxit:
        print("specs.shirley_calculate: Max iterations exceeded before convergence.")
        return 0
#     if is_reversed:
#         return ((yr + B)[::-1])
    else:
        return (yr + B)

def subtract_shirley_bg(xp : XPS_experiment, region : str, maxit : int = 10, lb : str = None, ax = None) -> XPS_experiment:
    """Plot region and shirley background. Decorator for shirley_loop function"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values
    col = plot_region(xp = xp, region = region, lb = lb, ax = ax).get_color()

    find_integration_limits(x, y, flag_plot=True, region = region, ax = ax)
    ybg = shirley_loop(x, y, maxit = maxit)

    if ax == None: ax = plt.gca()
    ax.plot(x, ybg, '--', color=col, label='__nolabel__');
    #cosmetics_plot(ax = ax)

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y - ybg})
    xpNew = deepcopy(xp)
    xpNew.dfx[region] = dfnew
    return xpNew

def subtract_double_shirley(xp : XPS_experiment, region : str, xlim : float, maxit : int = 10, lb : str = None, ax = None) -> XPS_experiment:
    """Shirley bg subtraction for double peak"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values
    col = plot_region(xp, region, ax = ax, lb=lb).get_color()

    y1 = y[ x > xlim ]
    x1 = x[ x > xlim ]
    y2 = y[ x <= xlim ]
    x2 = x[ x <= xlim ]

    ybg1 = shirley_loop(x1, y1, maxit = maxit)
    ybg2 = shirley_loop(x2, y2, maxit = maxit)

    if ax == None: ax = plt.gca()
    ax.plot(x, np.append(ybg1, ybg2), '--', color=col, label='__nolabel__')
    y12 = np.append( y1 - ybg1, y2 - ybg2)

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y12})
    xpNew = deepcopy(xp)
    xpNew.dfx[region] = dfnew
    return xpNew

def subtract_linear_bg (xp : XPS_experiment, region, lb : str = None, ax = None) -> XPS_experiment:
    """Fit background to line and subtract from data"""

    from scipy import stats, polyval
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values
    col = plot_region(xp, region, lb=region, ax=ax).get_color()

    bl = peakutils.baseline(y, deg=1)
    if ax == None: ax = plt.gca()
    ax.plot(x, bl, '--', color=col, label='Linear Background')

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y - bl})
    xpNew = deepcopy(xp)
    xpNew.dfx[region] = dfnew
    return xpNew

###########################   To use in nb with list of experiments   ###########################
def bulk_bg_subtract(experiments : list, regions : list) -> list:
    """Perform shirley bg subtraction on specified regions from several experiments
    Plot results and store them new list of experiments"""
    bg_exps = []
    fig, ax = plt.subplots(len(regions),2, figsize=(12, 10 * len(regions)))
    for xp in experiments:
        xp_bg = deepcopy(xp)   # Store current state of xp
        for i,r in enumerate(regions):
    #         print(xp.name, r)
            try:
                xp_r = subtract_shirley_bg(xp, r, maxit=40, lb=xp.label, ax=ax[i][0]);    # Perform bg subtraction
            except AssertionError:
                print('Max iterations exceeded, subtract linear baseline')
                xp_bg = subtract_linear_bg(xp, r, lb=xp.label, ax = ax[i][0])
            plot_region(xp_r, r, ax=ax[i][1])
            ax[i][0].set_title(r)
            ax[i][1].set_title('Subtraction result')

            cosmetics_plot(ax=ax[i][0], leg = False)
            cosmetics_plot(ax=ax[i][1])

            xp_bg.dfx[r] = xp_r.dfx[r]   # Store bg-subtracted region
        bg_exps.append(xp_bg)   # Store bg-subtracted regions experiment
    return bg_exps

def region_bg_subtract(experiments : list, region = str) -> list:
    """Inspect individual shirley bg subtraction for specified region from several experiments
    Plot results and store them new list of experiments"""
    bg_exps = []
    fig, ax = plt.subplots(len(experiments), 2, figsize=(12, 12 * len(regions)))
    for j, xp in enumerate(experiments):
        try:
            xp_bg = subtract_shirley_bg(xp, region, maxit=100, lb='__nolabel__', ax = ax[j][0])
        except AssertionError:
            print('Max iterations exceeded, subtract linear baseline')
            xp_bg = subtract_linear_bg(xp, region, lb='__nolabel__', ax = ax[j][0])
        bg_exps.append(xp_bg)
        plot_region(xp_bg, region, ax=ax[j][1])
        ax[j][0].set_title(xp.name)
        ax[j][1].set_title('Subtraction result')
        for i in range(2): cosmetics_plot(ax=ax[j][i])
    fig.tight_layout()
    return bg_exps
