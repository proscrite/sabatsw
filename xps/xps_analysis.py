import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import peakutils
from scipy.optimize import curve_fit

from dataclasses import dataclass
from xps.xps_import import XPS_experiment


def plot_region(xp : XPS_experiment, region : str, lb : str = None, ax = None):
    """Quick region plotter"""
    if lb == None: lb = xp.name + ', ' + region
    if ax == None: ax = plt.gca()
    p1 = ax.plot(xp.dfx[region].energy, xp.dfx[region].counts, label=lb)
    cosmetics_plot()
    return p1[0]

def cosmetics_plot(ax = None, leg : bool = True):
    if ax == None: ax = plt.gca()
    ax.invert_xaxis()
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('CPS [A.U.]')
    if leg: ax.legend()

def gaussian_smooth(xp : XPS_experiment, region, sigma : int = 2) -> XPS_experiment:
    from scipy.ndimage.filters import gaussian_filter1d

    y = gaussian_filter1d(xp.dfx[region].dropna().counts.values, sigma = 2)
    dfnew = pd.DataFrame({'energy' : xp.dfx[region].energy.dropna(), 'counts' : y})
    xp.dfx[region] = dfnew
    return xp

def flexible_integration_limits(xp : XPS_experiment, region : str, doublePeak : float = 0, flag_plot : bool = True) -> list:
    """Autolocate limits for area integration.
    doublePeak > 0 : second peak on the rhs of the main one
    doublePeak < 0 : second peak on the lhs of the main one
    doublePeak == 0 : no second peak
    Returns:
    --------
    [maxidx, (maxidx2), lmidx(2), rmidx(2)]
    Position index of [maxima, (second max), left minima, right minima]
    (2) denotes from secondary peak"""

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts
    plt.plot(x, y, label=xp.name)
    maxidx = abs(y - np.max(y)).idxmin()
    lmidx = abs(y[0:maxidx] - np.min(y[0:maxidx])).idxmin()
    rmidx = abs(y[maxidx:] - np.min(y[maxidx:])).idxmin() #+ maxidx

    if doublePeak < 0:
        maxidx2 = abs(y[:lmidx] - np.max(y[:lmidx])).idxmin()
        lmidx2 = abs(y[:maxidx2] - np.min(y[:maxidx2])).idxmin()
        ind = [maxidx, maxidx2, lmidx2, rmidx]
    elif doublePeak > 0:
        maxidx2 = abs(y[rmidx:] - np.max(y[rmidx:])).idxmin()
        rmidx2 = abs(y[maxidx2:] - np.min(y[maxidx2:])).idxmin()
        ind = [maxidx, maxidx2, lmidx, rmidx2]
    else:
        ind = [maxidx, lmidx, rmidx]

    ybase = plt.ylim()[0]
    for i in ind:
        plt.vlines(x[i], ymin=ybase, ymax=y[i], linestyles='--')
        plt.text(s='%.2f'%x[i], x = x[i], y = y[i])
    cosmetics_plot()
    return ind

def compare_areas(xp_ref : XPS_experiment, xp_sg : XPS_experiment, region : str,
                  lmidx : int, rmidx : int, lb : str = None,  ax = None):
    """Returns absolute and relative area in a region xp_sg and w.r.t. xp_ref
    between indices lmidx and rmidx"""
    y_ref = xp_ref.dfx[region].dropna().counts
    y_sg = xp_sg.dfx[region].dropna().counts

    if ax == None: ax = plt.gca()
    x = xp_sg.dfx[region].dropna().energy
    step = x[0] - x[1]

    area = np.trapz(y_sg [ lmidx : rmidx ], dx = step)
    area_rel = area / np.trapz(y_ref [ lmidx : rmidx ], dx = step)

    if lb == None: lb = xp_sg.name
    ax.plot(x, y_sg, '-', label=lb)
    ax.fill_between(x [lmidx : rmidx], y1 = y_sg[lmidx], y2 = y_sg [lmidx : rmidx], alpha=0.3)

    cosmetics_plot()
    return area_rel, area

def inset_rel_areas(area_rel : list, names : list) -> None:
    ax = plt.gca()
    axins = plt.axes([0.65, 0.5, 0.25, 0.3])
    axins.bar(names, area_rel)
    axins.set_ylabel('$A_{exp}/A_{ref}$', fontsize=12)
    axins.tick_params(labelrotation=45)
    ax.legend(loc='upper left')

def gauss(x, *p):
    A, mu, sigma = p
    return A *  np.exp(-( x-mu )**2 / (2.*sigma**2))

def double_gauss(x, *p):
    return gauss(x, *p[:3]) + gauss(x, *p[3:])

def compute_p0_peaks(x : np.array, y : np.array, thres0 : float, Npeaks : int) -> list:
    """Rough first estimation of fit parameters p0 from peak search"""

    peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
    while len(peaks) > Npeaks:
        thres0 += 0.05
        peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
    while len(peaks) < Npeaks:
        thres0 -= 0.05
        peaks = peakutils.indexes(y, thres = thres0, min_dist=10)

    p0 = []
    for i in range(Npeaks):
        xmax, ymax = x[peaks[i]], y[peaks[i]]
        p0.append(ymax)
        p0.append(xmax)
        p0.append(x[y > ymax/2][0] - xmax)
    return p0

def fit_double_gauss(xp : XPS_experiment, region : str, thres0 : float = 0.5):
    """Fit to double gauss, estimate loc and scale from peak finding"""

    col = plot_region(xp, region, lb=region).get_color()

    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values

    p0 = compute_p0_peaks(x, y, thres0, Npeaks=2)
    fit, cov = curve_fit(double_gauss, xdata = x , ydata= y, p0=p0)

    plt.plot(x, double_gauss(x, *fit), '--', color=col, label='Double gauss fit')

    plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)
    plt.text(s='%.1f'%fit[4], x=fit[4], y=fit[3]*1.05)
    yl = plt.ylim()
    plt.ylim(yl[0], yl[1]*1.5)

    return fit[1], fit[4]

def fit_gauss(xp : XPS_experiment, region : str, thres0 : float = 0.7):
    """Fit to gauss, estimate loc and scale from peak finding"""

    col = plot_region(xp, region, lb=region).get_color()
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values

    p0 = compute_p0_peaks(x, y, thres0, 1)
    fit, cov = curve_fit(self.gauss, x, y, p0=p0)

    plt.plot(x, self.gauss(x, *fit), '--', color=col, label='Gauss fit')

    plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)
    yl = plt.ylim()
    plt.ylim(yl[0], yl[1]*1.5)
    self.cosmetics_plot()

    return fit

def compute_gauss_area(fit, prefix):
    sigma = fit.best_values[prefix+'sigma']
    amp = fit.best_values[prefix+'amplitude']
    return amp * np.sqrt(np.pi/sigma)

def fit_voigt(xp : XPS_experiment, region : str, pars : list = None, bounds : list = None, lb : str = None, ax = None, flag_plot : bool = True):
    """General method for fitting voigt model
    Input
    ----------
    xp : class XPS_experiment
        XPS data
    region : str
        core level name
    pars, bounds : list
        initial guess of the fit parameters and bounds. If unspecified, guessed automatically
    Returns
    -----------
    fitv : lmfit.model
        fit result to Voigt model
    """
    from lmfit.models import PseudoVoigtModel, GaussianModel

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts

    col = plot_region(xp, region, lb, ax).get_color()

    mod = PseudoVoigtModel(prefix='v_')
    if pars == None:
        pars = mod.guess(y, x=x)
        pars['v_sigma'].set(value=1) # Usually guessed wrong anyway

    fitv = mod.fit(y, pars, x=x)

    if flag_plot:
        if ax == None : ax = plt.gca()
        ax.plot(x, fitv.best_fit, '--', label='Fit to single Voigt')
        ax.legend()
    return fitv

def add_gauss_shoulder(xp : XPS_experiment, region : str, par_g : list, bounds_g: list,
                       fitv = None, Ng : int = 1, lb : str = None, ax = None, flag_plot : bool = True):
    """Add gaussian shoulder to fit
    Input
    ----------
    xp : class XPS_experiment
        XPS data
    region : str
        core level name
    par_g, bounds_g : list
        initial guess of the gauss fit parameters and bounds.
    Returns
    -----------
    fitvg : lmfit.model
        fit result to Voigt + Gaussian model
    """
    from lmfit.models import PseudoVoigtModel, GaussianModel

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts
    if ax == None : ax = plt.gca()
    col = plot_region(xp, region, lb, ax).get_color()

    gauss2 = GaussianModel(prefix='g'+str(Ng)+'_')
    pars = fitv.params
    pars.update(gauss2.make_params())

    for k,p,b in zip(gauss2.param_names, par_g, bounds_g):
        pars[k].set(value=p, min=b[0], max=b[1])
    mod2 = fitv.model + gauss2

    fitvg = mod2.fit(y, pars, x=x)
    # print(fitvg.fit_report(min_correl=.5))
    if flag_plot:
        comps = fitvg.eval_components(x=x)
        a1, a0 = [compute_gauss_area(fitvg, prefix) for prefix in ['g'+str(Ng)+'_', 'v_']]

        ax.plot(x, fitvg.best_fit, '-r', label = 'best fit')
        ax.plot(x, comps['v_'], 'g--', label = 'voigt component @ %.2f \nArea ref. '%fitvg.best_values['v_center'])
        ax.plot(x, comps['g'+str(Ng)+'_'], 'y--', label = '1st gauss component @ %.2f \nArea =  %.2f \nArea ref'%(fitvg.best_values['g'+str(Ng)+'_center'], a1/a0))
        ax.legend()
    cosmetics_plot()
    return fitvg

def fit_double_voigt(xp : XPS_experiment, region : str, pars : list = None, bounds : list = None, sepPt : float = None,
                     lb : str = None, ax = None, flag_plot : bool = True, DEBUG : bool = False):
    """Fitting double voigt model
    Input
    ----------
    xp : class XPS_experiment
        XPS data
    region : str
        core level name
    pars, bounds : list
        initial guess of the fit parameters and bounds. If unspecified, guessed automatically
    sepPt : float
        separation point in energy between the two peaks. If unspecified guessed automatically
    flag_plot, DEBUG : bool
        flags to plot intermediate and final fit results
    Returns
    -----------
    fitv : lmfit.model
        fit result to Voigt model
    """
    from lmfit.models import PseudoVoigtModel

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts
    if sepPt == None: sepPt = find_separation_point(x, y)

    x1 = x[x<sepPt].values
    x2 = x[x>sepPt].values
    y1 = y[x<sepPt].values
    y2 = y[x>sepPt].values
    if ax == None : ax = plt.gca()

    col = plot_region(xp, region, lb=xp.name, ax=ax).get_color()

    mod1 = PseudoVoigtModel(prefix='v1_')
    mod2 = PseudoVoigtModel(prefix='v2_')
    if pars == None:
        pars1 = mod1.guess(y1, x=x1)
        pars1['v1_sigma'].set(value=1) # Usually guessed wrong anyway
        pars2 = mod2.guess(y2, x=x2)
        pars2['v2_sigma'].set(value=1) # Usually guessed wrong anyway

    mod = mod1 + mod2
    pars = mod.make_params()
    pars.update(pars1)
    pars.update(pars2)
    if DEBUG:
        fit1 = mod1.fit(y1, x=x1, params=pars1)
        fit2 = mod2.fit(y2, x=x2, params=pars2)
        ax.plot(x1, fit1.best_fit, '--', label='Fit first Voigt')
        ax.plot(x2, fit2.best_fit, '--', label='Fit second Voigt')

    fitv = mod.fit(y, pars, x=x)

    if flag_plot:
        ax.plot(x, fitv.best_fit, '--', label='Fit to double Voigt')
        ax.legend()

    return fitv

def find_separation_point(x : np.array, y : np.array, min_dist : int = 20,
                          ax = None, DEBUG : bool = False) -> float:
    """Autolocate separation point between two peaks for double fitting"""
    peaks = [0, 0, 0]
    thres = 0.8
    while len(peaks) > 2:
        peaks = peakutils.indexes(y, thres=thres, min_dist=min_dist)
        thres += 0.01
    if DEBUG:
        if ax == None : ax = plt.gca()
        ax.plot(x[peaks], y[peaks], '*', ms=10)
        ax.axvline(x[peaks].sum()/2)
    return x[peaks].sum()/2
