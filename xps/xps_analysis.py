import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import peakutils
import re

from copy import deepcopy
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

def trim_spectra(xp : XPS_experiment, xpRef : XPS_experiment, region, inplace : bool = False) -> XPS_experiment:
    """Crop spectra with different bounds so that they coincide with xpRef
    xpRef should have the shortest spectrum on both ends"""
    eup = xpRef.dfx[region].energy.head(1)
    edw = xpRef.dfx[region].dropna().energy.tail(1)

    cutup = np.where(xp.dfx[region].energy.values > eup.values)[0]
    cutdown = np.where(xp.dfx[region].dropna().energy.values < edw.values)[0]

    dfnew = xp.dfx[region].dropna().drop(cutup).drop(cutdown).reset_index(drop=True)

    if inplace:
        xp.dfx[region] = dfnew
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
    return xpNew

def crop_spectrum(xp : XPS_experiment, region : str,
                  eup : float = None, edw : float = None, inplace : bool = False):
    """Manually bound region spectrum to upper (eup) and lower (edw) limits """
    if eup == None: eup = xp.dfx[region].energy.head(1)[0]
    if edw == None: edw = xp.dfx[region].dropna().energy.tail(1).values[0]

    dropup = np.where(xp.dfx[region].energy.values > eup)[0]
    dfup = xp.dfx[region].dropna().drop(dropup).reset_index(drop=True)

    dropdw = np.where(dfup.energy.values < edw)[0]
    dfnew = dfup.drop(dropdw).reset_index(drop=True)

    if inplace:
        xp.dfx[region] = dfnew
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
        return xpNew

def trim_regions(experiments: list, regions: list)->list:
    """Loop over specified regions, locate the experiment with shortest ends
        and trim all spectra to meet those bounds"""

    ### First find shortest spectra on both ends:
    for i,r in enumerate(regions):
        boundUp, boundDw = [], []
        for xp in experiments:
            x = xp.dfx[r].dropna().energy.values
            boundUp.append(x[0])
            boundDw.append(x[-1])
        idBoundUp = np.argmin(boundUp)
        idBoundDw = np.argmax(boundDw)

    ### Now trim each
    fig, ax = plt.subplots(len(regions), figsize=(8, 8*len(regions)))
    trimmed_exps = []
    for xp in experiments:
        xp_trim = deepcopy(xp)
        for i,r in enumerate(regions):
            trim = trim_spectra(xp, xpRef=experiments[idBoundUp], region=r)   # Trim the upper ends
            trim = trim_spectra(trim, xpRef=experiments[idBoundDw], region=r) # Trim the lower ends
            plot_region(trim, r, ax=ax[i], lb=trim.name)
            ax[i].set_title(r)
            ax[i].legend()
            xp_trim.dfx[r] = trim.dfx[r]
        trimmed_exps.append(xp_trim)
    return trimmed_exps

def gaussian_smooth(xp : XPS_experiment, region, sigma : int = 2) -> XPS_experiment:
    from scipy.ndimage.filters import gaussian_filter1d

    y = gaussian_filter1d(xp.dfx[region].dropna().counts.values, sigma = 2)
    dfnew = pd.DataFrame({'energy' : xp.dfx[region].energy.dropna(), 'counts' : y})

    xpNew = deepcopy(xp)
    xpNew.dfx[region] = dfnew
    return xpNew

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
                  lmidx : int, rmidx : int, lb : str = None,  ax = None, flag_fill : bool = False):
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
    if flag_fill:
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

def fit_voigt(xp : XPS_experiment, region : str,
              pars : list = None, bounds : list = None, prefix : str = 'v_',
              lb : str = None, ax = None, flag_plot : bool = True):
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

    mod = PseudoVoigtModel(prefix=prefix)
    if pars == None:
        pars = mod.guess(y, x=x)
        pars[prefix+'sigma'].set(value=1) # Usually guessed wrong anyway
        pars[prefix+'fraction'].set(value=0.2, min=0.15, max=0.20)

    fitv = mod.fit(y, pars, x=x)

    if flag_plot:
        if ax == None : ax = plt.gca()
        ax.plot(x, fitv.best_fit, '--', label='Voigt fit, $\chi^2_N$ = %i' %fitv.redchi)
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

        ax.plot(x, fitvg.best_fit, '-r', label = 'best fit, $\chi^2_N$ = %i' %fitvg.redchi)
        for compo in comps:
            colc = ax.plot(x, comps[compo], ls='dashdot', label = '%scenter: %.2f' %(compo, fitvg.best_values[compo+'center']) )[0].get_color()
            ax.fill_between(x, y1 = 0, y2 = comps[compo], alpha=0.3, color=colc)
        ax.legend()
    cosmetics_plot(ax=ax)
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
                          thres0 :float = 0.5, ax = None, DEBUG : bool = False) -> float:
    """Autolocate separation point between two peaks for double fitting"""
    peaks = [0, 0, 0]
    thres = thres0
    while len(peaks) > 2:
        peaks = peakutils.indexes(y, thres=thres, min_dist=min_dist)
        thres += 0.01
    if DEBUG:
        if ax == None : ax = plt.gca()
        ax.plot(x[peaks], y[peaks], '*', ms=10)
        ax.axvline(x[peaks].sum()/2)
    return x[peaks].sum()/2

def check_pars_amplitud(pars, prefix : str, x : np.array, y : np.array):
    if pars[prefix + 'amplitude'] < 0 :
        amp = y[np.where(x == pars[prefix + 'center'].value)[0][0]]
        pars[prefix + 'amplitude'].set(value=amp)
    return pars

def fit_double_shouldered_voigt(xp : XPS_experiment, region : str, par_g1 : list, bound_g1 : list,
                               par_g2 : list, bound_g2 : list, lb : str = None,
                                ax = None, flag_plot : bool = True) -> tuple:

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts
    ### Separate two peaks ###
    sepPt = find_separation_point(x, y)
    x1 = x[x<sepPt].values
    x2 = x[x>sepPt].values
    y1 = y[x<sepPt].values
    y2 = y[x>sepPt].values

    ### Set and guess main (voigt) components  ####
    vo1 = PseudoVoigtModel(prefix='v1_')
    pars1 = vo1.guess(y1, x=x1)
    vo2 = PseudoVoigtModel(prefix='v2_')
    pars2 = vo2.guess(y2, x=x2)
    pars1['v1_sigma'].set(value=1) # Usually guessed wrong anyway
    pars1['v1_fraction'].set(value=0.15, min=0.15, max=0.2)
    pars2['v2_sigma'].set(value=1) # Usually guessed wrong anyway
    pars2['v2_fraction'].set(value=0.15, min=0.15, max=0.2)

    ### Set gaussian shoulders ###
    gauss1 = GaussianModel(prefix='g1_')
    pars1.update(gauss1.make_params())

    gauss2 = GaussianModel(prefix='g2_')
    pars2.update(gauss2.make_params())
    mod1 = vo1 + gauss1
    mod2 = vo2 + gauss2

    for k,p,b in zip(gauss1.param_names, par_g1, bounds_g1):
        pars1[k].set(value=p, min=b[0], max=b[1])

    for k,p,b in zip(gauss2.param_names, par_g2, bounds_g2):
        pars2[k].set(value=p, min=b[0], max=b[1])

    fitvg1 = mod1.fit(y1, pars1, x=x1)
    fitvg2 = mod2.fit(y2, pars2, x=x2)

    if ax == None: ax = plt.gca()
    col = plot_region(xp, region, lb, ax).get_color()
    comps1 = fitvg1.eval_components(x=x1)
    ax.plot(x1, fitvg1.best_fit, '-r', label = 'best fit, $\chi^2_N$ = %i' %fitvg1.redchi)
    for compo in comps1:
        colc = ax.plot(x1, comps1[compo], ls='dashdot', label = '%scenter: %.2f' %(compo, fitvg1.best_values[compo+'center']) )[0].get_color()
        ax.fill_between(x1, y1 = 0, y2 = comps1[compo], alpha=0.3, color=colc)

    comps2 = fitvg2.eval_components(x=x2)
    ax.plot(x2, fitvg2.best_fit, '-', label = 'best fit, $\chi^2_N$ = %i' %fitvg2.redchi)
    for compo in comps2:
        colc = ax.plot(x2, comps2[compo], ls='dashdot', label = '%scenter: %.2f' %(compo, fitvg2.best_values[compo+'center']) )[0].get_color()
        ax.fill_between(x2, y1 = 0, y2 = comps2[compo], alpha=0.3, color=colc)

    ax.legend()
    return fitvg1, fitvg2

############# Stoichiometry and fit table utilities  ####################

def make_header(num : list, denom : list):
    head = 'Experiment\t'
    for n, d in zip(num, denom):
        cln = re.search(r'\d+', n).span()[0]
        cld = re.search(r'\d+', d).span()[0]
        head += n[:cln] + '/' + d[:cld] + '\t'
    print(head)

def regionPairs2numDenom(pairs : tuple):
    """Reshape a tuple of region pairs (ex. N/C, Br/O)
    into (num, denom) tuple (ex. ('N1s', 'C1s'), ('Br3p', 'O1s'))"""
    transpose = np.array(pairs).T
    assert transpose.shape[0] == 2, "Passed tuple has incorrect shape"
    num, denom = transpose
    return num, denom

def make_stoichometry_table(exps : list, num : list, denom : list):
    """Print stoichiometry table of the experiments exps at the regions in num/denom
    Example: table_print(oxid_exps, ('N1s', 'C1s'), ('Br3p', 'O1s'))
    will print the stoichiometry N/C, Br/O for the passed experiments"""
    make_header(num = num, denom = denom)
#     print('Experiment, ' + )
    for i, xp in enumerate(exps):
        row = xp.name + '\t'
        for i, j in zip (num, denom):
            row += ('%.2f\t ' %(xp.area[i]/xp.area[j]))
        print(row )

def component_areas(fit, x : np.array = None) -> tuple:
    """Numerical integration of area components of lmfit.result
    Arguments:
    fit : lmfit.Model.result
    x : np.array
        Independendt variable, if unspecified, dx = 1 for numerical integration
    Returns:
    rel_areas : dict
        Component areas normalized to total sum
    areas : dict
        keys are component prefixes and values the integrated areas"""
    if x == None: dx = 1
    else: dx = x[0] - x[1]

    areas, rel_areas = {}, {}
    for key, val in zip(fit.eval_components().keys(), fit.eval_components().values()):
        areas.update({key+'area' : np.trapz(val, dx = dx)})
    for key, val in zip(areas.keys(), areas.values()):
        rel_areas.update({key : val/sum(areas.values())})

    return rel_areas, areas

def make_fit_table(fit, area : dict):
    """Print a table with fit results and relative areas dict"""
    par_table = ['center', 'fwhm', 'area']
    head = 'component \t'
    for par in par_table: head += '%s\t'%par
    print(head)

    for i, comp in enumerate(fit.components):
        pref = comp.prefix
        line = pref + '\t\t'
        for par in par_table[:-1]:
            line += '%.2f \t'%fit.values[pref + par]
        line += '%.2f \t'%area[pref+'area']
        print(line)

def barplot_fit_fwhm(experiments : list, fit : np.array):
    names = [xp.name for xp in experiments]

    colv = plt.errorbar(x = fit[:,0], y = names, xerr=fit[:,1]/2, fmt='o', mew=2, label='Main component')[0].get_color()

    dif = fit [:-1,0] - fit[1:,0]
    for i, d in enumerate(dif) :
        plt.annotate(s = '$\Delta E = $%.2f'%d, xy=(fit[i+1,0], 0.8 * (i+1)), color=colv)
        plt.fill_betweenx(y=(i, i+1), x1=fit[i,0], x2=fit[i+1,0], alpha=0.3, color=colv)

    if fit.shape[1] > 2:
        colg = plt.errorbar(x = fit[:,2], y = names, xerr=fit[:,3]/2, fmt='o', mew=2,label='Shoulder')[0].get_color()
        difg = fit [:-1,2] - fit[1:,2]
        for i, d in enumerate(difg) :
            plt.annotate(s = '$\Delta E = $%.2f'%d, xy=(fit[i+1,2], 0.8 * (i+1)), color=colg)
            plt.fill_betweenx(y=(i, i+1), x1=fit[i,2], x2=fit[i+1,2], alpha=0.3, color=colg)
    cosmetics_plot()
    plt.ylabel('')

def plot_xp_regions(experiments : list, regions : list, colors : list = None):
    """Subplots all regions of a list of experiments (unnormalised)"""
    rows = int(np.ceil(len(regions) / 3))
    cols = 3

    fig, ax = plt.subplots(rows, cols, figsize=(16, 8))
    for i,r in enumerate(regions):
        for c,xp in enumerate(experiments):
            j, k = i//3, i%3
            if i == len(regions) - 1:   # Set labels from last region
                li = plot_region(xp, r, ax=ax[j][k], lb=xp.name)
                if len(colors) > 0: li.set_color(colors[c])
                ax[j][k].set_title('Au_4f')
                ax[j][k].get_legend().remove()
            else:
                li = plot_region(xp, r, ax=ax[j][k], lb='__nolabel__')
                if len(colors) > 0: li.set_color(colors[c])
                ax[j][k].set_title(r)
            cosmetics_plot(ax=ax[j][k], leg = False);
        if len(experiments)%2 == 0:
            ax[j][k].invert_xaxis()
    plt.tight_layout()
    fig.legend()
