import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import peakutils
from scipy.optimize import curve_fit

from dataclasses import dataclass
from xps.xps_import import XPS_experiment


def plot_region(xp : XPS_experiment, region : str, lb : str = None):
    """Quick region plotter"""
    if lb == None: lb = xp.name + region

    p1 = plt.plot(xp.dfx[region].energy, xp.dfx[region].counts, label=lb)
    cosmetics_plot()
    return p1[0]

def cosmetics_plot(ax = None):
    if ax == None: ax = plt.gca()
    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('CPS [A.U.]')

def gaussian_smooth(xp : XPS_experiment, region, sigma : int = 2) -> XPS_experiment:
    from scipy.ndimage.filters import gaussian_filter1d

    y = gaussian_filter1d(xp.dfx[region].dropna().counts.values, sigma = 2)
    dfnew = pd.DataFrame({'energy' : xp.dfx[region].energy.dropna(), 'counts' : y})
    xp.dfx[region] = dfnew
    return xp

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

def fit_voigt(xp : XPS_experiment, region : str, pars : list = None, bounds : list = None, ax = None, flag_plot : bool = True):
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
    if ax == None: ax = plt.gca()
    ax.plot(x, y, '-b', label='Data')

    mod = PseudoVoigtModel(prefix='v_')
    if pars == None:
        pars = mod.guess(y, x=x)
        pars['v_sigma'].set(value=1) # Usually guessed wrong anyway

    fitv = mod.fit(y, pars, x=x)

    if flag_plot:
        ax.plot(x, fitv.best_fit, '--', label='Fit to single Voigt')
        ax.legend()
    return fitv

def add_gauss_shoulder(xp : XPS_experiment, region : str, par_g : list, bounds_g: list,
                       fitv = None, Ng : int = 1, ax = None, flag_plot : bool = True):
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
    ax.plot(x, y, '-b', label='Data')

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
#         a1, a0 = [compute_area(fitvg, prefix) for prefix in ['g_', 'v_']]

        ax.plot(x, fitvg.best_fit, '-r', label = 'best fit')
#         ax.plot(x, comps['v_'], 'g--', label = 'voigt component @ %.2f \nArea ref. '%fitvg.best_values['v_center'])
        ax.plot(x, comps['g'+str(Ng)+'_'], 'y--', label = '1st gauss component @ %.2f \nArea =  '%(fitvg.best_values['g'+str(Ng)+'_center']))#, a1/a0))
        ax.legend()
    cosmetics_plot()
    return fitvg
