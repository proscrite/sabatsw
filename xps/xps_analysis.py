import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from scipy.optimize import curve_fit

from dataclasses import dataclass
from xps.xps_import import XPS_experiment

def find_integration_limits(x, y, flag_plot = False, region : str = None):
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
        plt.plot(x, y, 'b-', label=region.replace('_', ' '))
        ybase = plt.ylim()[0]
        ind = [maxidx, lmidx, rmidx]
        for i in ind:
            plt.vlines(x = x[i], ymin=ybase, ymax=y[i], color='k')
            plt.text(s= '%.2f'%x[i], x = x[i], y = y[i])

    return lmidx, rmidx

def shirley_loop(x, y,
                 lmidx : int = None,
                 rmidx : int = None,
                 maxit : int = 10, tol : float = 1e-5,
                 DEBUG : bool = False):
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

    if it >= maxit:
        print("specs.shirley_calculate: Max iterations exceeded before convergence.")
#     if is_reversed:
#         return ((yr + B)[::-1])
    else:
        return (yr + B)

def subtract_shirley_bg(xp : XPS_experiment, region : str, maxit : int = 10, lb : str = None) -> XPS_experiment:
    """Plot region and shirley background. Decorator for shirley_loop function"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values

#     if lb == None : lb = region.replace('_', ' ')
    p1 = plt.plot(x, y, label='')
    col = p1[0].get_color()

    find_integration_limits(x, y, flag_plot=True, region = region)
    ybg = shirley_loop(x, y, maxit = maxit)

    plt.plot(x, ybg, '--',  label='Shirley Background')
    cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : dfx.energy.dropna(), 'counts' : y - ybg})
    xp.dfx[region] = dfnew
    return xp

def subtract_double_shirley(xp : XPS_experiment, region : str, xlim : float, maxit : int = 10) -> XPS_experiment:
    """Shirley bg subtraction for double peak"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values

    plt.plot(x, y, label=region.replace('_', ' '))

    y1 = y[ x > xlim ]
    x1 = x[ x > xlim ]
    y2 = y[ x <= xlim ]
    x2 = x[ x <= xlim ]

    ybg1 = shirley_loop(x1, y1, maxit = maxit)
    ybg2 = shirley_loop(x2, y2, maxit = maxit)

    plt.plot(x, np.append(ybg1, ybg2), label='Double shirley bg')
    y12 = np.append( y1 - ybg1, y2 - ybg2)
#     cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y12})
    xp.dfx[region] = dfnew
    return xp

def subtract_linear_bg (xp : XPS_experiment, region, lb : str = None) -> XPS_experiment:
    """Fit background to line and subtract from data"""

    from scipy import stats, polyval
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values
    if lb == None : lb = region.replace('_', ' ')
    p1 = plt.plot(x, y, label=lb)
    col = p1[0].get_color()

    slope, intercept, r, p_val, std_err = stats.linregress(x, y)
    ybg = polyval([slope, intercept], x);
    plt.plot(x, ybg, '--', color=col, label='Linear Background')
#     cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : self.df[region].energy.dropna(), 'counts' : y - ybg})
    xp.dfx[region] = dfnew
    return xp

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

#     p1 = plot_region(region, lb=region)
#     col = p1.get_color()
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values

    p0 = compute_p0_peaks(x, y, thres0, Npeaks=2)
    fit, cov = curve_fit(double_gauss, xdata = x , ydata= y, p0=p0)

    plt.plot(x, double_gauss(x, *fit), '--', )#color=col, label='Double gauss fit')

    plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)
    plt.text(s='%.1f'%fit[4], x=fit[4], y=fit[3]*1.05)
    yl = plt.ylim()
    plt.ylim(yl[0], yl[1]*1.5)
    cosmetics_plot()

    return fit[1], fit[4]

def fit_gauss(xp : XPS_experiment, region : str, thres0 : float = 0.7):
    """Fit to gauss, estimate loc and scale from peak finding"""

#     p1 = plot_region(region, lb=region)
#     col = p1.get_color()
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


def plot_region(xp : XPS_experiment, region : str, lb : str = None):
    """Quick region plotter"""
    if lb == None: lb = xp.other_meta

    p1 = plt.plot(xp.dfx[region].energy, xp.dfx[region].counts, label=lb)
    cosmetics_plot()
    return p1[0]

def cosmetics_plot(ax = None):
    if ax == None: ax = plt.gca()
    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('CPS [A.U.]')def find_integration_limits(x, y, flag_plot = False, region : str = None):
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
        plt.plot(x, y, 'b-', label=region.replace('_', ' '))
        ybase = plt.ylim()[0]
        ind = [maxidx, lmidx, rmidx]
        for i in ind:
            plt.vlines(x = x[i], ymin=ybase, ymax=y[i], color='k')
            plt.text(s= '%.2f'%x[i], x = x[i], y = y[i])

    return lmidx, rmidx

def shirley_loop(x, y,
                 lmidx : int = None,
                 rmidx : int = None,
                 maxit : int = 10, tol : float = 1e-5,
                 DEBUG : bool = False):
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

    if it >= maxit:
        print("specs.shirley_calculate: Max iterations exceeded before convergence.")
#     if is_reversed:
#         return ((yr + B)[::-1])
    else:
        return (yr + B)

def subtract_shirley_bg(xp : XPS_experiment, region : str, maxit : int = 10, lb : str = None) -> XPS_experiment:
    """Plot region and shirley background. Decorator for shirley_loop function"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values

#     if lb == None : lb = region.replace('_', ' ')
    p1 = plt.plot(x, y, label='')
    col = p1[0].get_color()

    find_integration_limits(x, y, flag_plot=True, region = region)
    ybg = shirley_loop(x, y, maxit = maxit)

    plt.plot(x, ybg, '--',  label='Shirley Background')
    cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : dfx.energy.dropna(), 'counts' : y - ybg})
    xp.dfx[region] = dfnew
    return xp

def subtract_double_shirley(xp : XPS_experiment, region : str, xlim : float, maxit : int = 10) -> XPS_experiment:
    """Shirley bg subtraction for double peak"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values

    plt.plot(x, y, label=region.replace('_', ' '))

    y1 = y[ x > xlim ]
    x1 = x[ x > xlim ]
    y2 = y[ x <= xlim ]
    x2 = x[ x <= xlim ]

    ybg1 = shirley_loop(x1, y1, maxit = maxit)
    ybg2 = shirley_loop(x2, y2, maxit = maxit)

    plt.plot(x, np.append(ybg1, ybg2), label='Double shirley bg')
    y12 = np.append( y1 - ybg1, y2 - ybg2)
#     cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y12})
    xp.dfx[region] = dfnew
    return xp

def subtract_linear_bg (xp : XPS_experiment, region, lb : str = None) -> XPS_experiment:
    """Fit background to line and subtract from data"""

    from scipy import stats, polyval
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values
    if lb == None : lb = region.replace('_', ' ')
    p1 = plt.plot(x, y, label=lb)
    col = p1[0].get_color()

    slope, intercept, r, p_val, std_err = stats.linregress(x, y)
    ybg = polyval([slope, intercept], x);
    plt.plot(x, ybg, '--', color=col, label='Linear Background')
#     cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : self.df[region].energy.dropna(), 'counts' : y - ybg})
    xp.dfx[region] = dfnew
    return xp

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

#     p1 = plot_region(region, lb=region)
#     col = p1.get_color()
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values

    p0 = compute_p0_peaks(x, y, thres0, Npeaks=2)
    fit, cov = curve_fit(double_gauss, xdata = x , ydata= y, p0=p0)

    plt.plot(x, double_gauss(x, *fit), '--', )#color=col, label='Double gauss fit')

    plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)
    plt.text(s='%.1f'%fit[4], x=fit[4], y=fit[3]*1.05)
    yl = plt.ylim()
    plt.ylim(yl[0], yl[1]*1.5)
    cosmetics_plot()

    return fit[1], fit[4]

def fit_gauss(xp : XPS_experiment, region : str, thres0 : float = 0.7):
    """Fit to gauss, estimate loc and scale from peak finding"""

#     p1 = plot_region(region, lb=region)
#     col = p1.get_color()
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


def plot_region(xp : XPS_experiment, region : str, lb : str = None):
    """Quick region plotter"""
    if lb == None: lb = xp.other_meta

    p1 = plt.plot(xp.dfx[region].energy, xp.dfx[region].counts, label=lb)
    cosmetics_plot()
    return p1[0]

def cosmetics_plot(ax = None):
    if ax == None: ax = plt.gca()
    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('CPS [A.U.]')def find_integration_limits(x, y, flag_plot = False, region : str = None):
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
        plt.plot(x, y, 'b-', label=region.replace('_', ' '))
        ybase = plt.ylim()[0]
        ind = [maxidx, lmidx, rmidx]
        for i in ind:
            plt.vlines(x = x[i], ymin=ybase, ymax=y[i], color='k')
            plt.text(s= '%.2f'%x[i], x = x[i], y = y[i])

    return lmidx, rmidx

def shirley_loop(x, y,
                 lmidx : int = None,
                 rmidx : int = None,
                 maxit : int = 10, tol : float = 1e-5,
                 DEBUG : bool = False):
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

    if it >= maxit:
        print("specs.shirley_calculate: Max iterations exceeded before convergence.")
#     if is_reversed:
#         return ((yr + B)[::-1])
    else:
        return (yr + B)

def subtract_shirley_bg(xp : XPS_experiment, region : str, maxit : int = 10, lb : str = None) -> XPS_experiment:
    """Plot region and shirley background. Decorator for shirley_loop function"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values

#     if lb == None : lb = region.replace('_', ' ')
    p1 = plt.plot(x, y, label='')
    col = p1[0].get_color()

    find_integration_limits(x, y, flag_plot=True, region = region)
    ybg = shirley_loop(x, y, maxit = maxit)

    plt.plot(x, ybg, '--',  label='Shirley Background')
    cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : dfx.energy.dropna(), 'counts' : y - ybg})
    xp.dfx[region] = dfnew
    return xp

def subtract_double_shirley(xp : XPS_experiment, region : str, xlim : float, maxit : int = 10) -> XPS_experiment:
    """Shirley bg subtraction for double peak"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values

    plt.plot(x, y, label=region.replace('_', ' '))

    y1 = y[ x > xlim ]
    x1 = x[ x > xlim ]
    y2 = y[ x <= xlim ]
    x2 = x[ x <= xlim ]

    ybg1 = shirley_loop(x1, y1, maxit = maxit)
    ybg2 = shirley_loop(x2, y2, maxit = maxit)

    plt.plot(x, np.append(ybg1, ybg2), label='Double shirley bg')
    y12 = np.append( y1 - ybg1, y2 - ybg2)
#     cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y12})
    xp.dfx[region] = dfnew
    return xp

def subtract_linear_bg (xp : XPS_experiment, region, lb : str = None) -> XPS_experiment:
    """Fit background to line and subtract from data"""

    from scipy import stats, polyval
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values
    if lb == None : lb = region.replace('_', ' ')
    p1 = plt.plot(x, y, label=lb)
    col = p1[0].get_color()

    slope, intercept, r, p_val, std_err = stats.linregress(x, y)
    ybg = polyval([slope, intercept], x);
    plt.plot(x, ybg, '--', color=col, label='Linear Background')
#     cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : self.df[region].energy.dropna(), 'counts' : y - ybg})
    xp.dfx[region] = dfnew
    return xp

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

#     p1 = plot_region(region, lb=region)
#     col = p1.get_color()
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values

    p0 = compute_p0_peaks(x, y, thres0, Npeaks=2)
    fit, cov = curve_fit(double_gauss, xdata = x , ydata= y, p0=p0)

    plt.plot(x, double_gauss(x, *fit), '--', )#color=col, label='Double gauss fit')

    plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)
    plt.text(s='%.1f'%fit[4], x=fit[4], y=fit[3]*1.05)
    yl = plt.ylim()
    plt.ylim(yl[0], yl[1]*1.5)
    cosmetics_plot()

    return fit[1], fit[4]

def fit_gauss(xp : XPS_experiment, region : str, thres0 : float = 0.7):
    """Fit to gauss, estimate loc and scale from peak finding"""

#     p1 = plot_region(region, lb=region)
#     col = p1.get_color()
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


def plot_region(xp : XPS_experiment, region : str, lb : str = None):
    """Quick region plotter"""
    if lb == None: lb = xp.other_meta

    p1 = plt.plot(xp.dfx[region].energy, xp.dfx[region].counts, label=lb)
    cosmetics_plot()
    return p1[0]

def cosmetics_plot(ax = None):
    if ax == None: ax = plt.gca()
    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('CPS [A.U.]')def find_integration_limits(x, y, flag_plot = False, region : str = None):
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
        plt.plot(x, y, 'b-', label=region.replace('_', ' '))
        ybase = plt.ylim()[0]
        ind = [maxidx, lmidx, rmidx]
        for i in ind:
            plt.vlines(x = x[i], ymin=ybase, ymax=y[i], color='k')
            plt.text(s= '%.2f'%x[i], x = x[i], y = y[i])

    return lmidx, rmidx

def shirley_loop(x, y,
                 lmidx : int = None,
                 rmidx : int = None,
                 maxit : int = 10, tol : float = 1e-5,
                 DEBUG : bool = False):
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

    if it >= maxit:
        print("specs.shirley_calculate: Max iterations exceeded before convergence.")
#     if is_reversed:
#         return ((yr + B)[::-1])
    else:
        return (yr + B)

def subtract_shirley_bg(xp : XPS_experiment, region : str, maxit : int = 10, lb : str = None) -> XPS_experiment:
    """Plot region and shirley background. Decorator for shirley_loop function"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values

#     if lb == None : lb = region.replace('_', ' ')
    p1 = plt.plot(x, y, label='')
    col = p1[0].get_color()

    find_integration_limits(x, y, flag_plot=True, region = region)
    ybg = shirley_loop(x, y, maxit = maxit)

    plt.plot(x, ybg, '--',  label='Shirley Background')
    cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : dfx.energy.dropna(), 'counts' : y - ybg})
    xp.dfx[region] = dfnew
    return xp

def subtract_double_shirley(xp : XPS_experiment, region : str, xlim : float, maxit : int = 10) -> XPS_experiment:
    """Shirley bg subtraction for double peak"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values

    plt.plot(x, y, label=region.replace('_', ' '))

    y1 = y[ x > xlim ]
    x1 = x[ x > xlim ]
    y2 = y[ x <= xlim ]
    x2 = x[ x <= xlim ]

    ybg1 = shirley_loop(x1, y1, maxit = maxit)
    ybg2 = shirley_loop(x2, y2, maxit = maxit)

    plt.plot(x, np.append(ybg1, ybg2), label='Double shirley bg')
    y12 = np.append( y1 - ybg1, y2 - ybg2)
#     cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y12})
    xp.dfx[region] = dfnew
    return xp

def subtract_linear_bg (xp : XPS_experiment, region, lb : str = None) -> XPS_experiment:
    """Fit background to line and subtract from data"""

    from scipy import stats, polyval
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values
    if lb == None : lb = region.replace('_', ' ')
    p1 = plt.plot(x, y, label=lb)
    col = p1[0].get_color()

    slope, intercept, r, p_val, std_err = stats.linregress(x, y)
    ybg = polyval([slope, intercept], x);
    plt.plot(x, ybg, '--', color=col, label='Linear Background')
#     cosmetics_plot()

    dfnew = pd.DataFrame({'energy' : self.df[region].energy.dropna(), 'counts' : y - ybg})
    xp.dfx[region] = dfnew
    return xp

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

#     p1 = plot_region(region, lb=region)
#     col = p1.get_color()
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values

    p0 = compute_p0_peaks(x, y, thres0, Npeaks=2)
    fit, cov = curve_fit(double_gauss, xdata = x , ydata= y, p0=p0)

    plt.plot(x, double_gauss(x, *fit), '--', )#color=col, label='Double gauss fit')

    plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)
    plt.text(s='%.1f'%fit[4], x=fit[4], y=fit[3]*1.05)
    yl = plt.ylim()
    plt.ylim(yl[0], yl[1]*1.5)
    cosmetics_plot()

    return fit[1], fit[4]

def fit_gauss(xp : XPS_experiment, region : str, thres0 : float = 0.7):
    """Fit to gauss, estimate loc and scale from peak finding"""

#     p1 = plot_region(region, lb=region)
#     col = p1.get_color()
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


def plot_region(xp : XPS_experiment, region : str, lb : str = None):
    """Quick region plotter"""
    if lb == None: lb = xp.other_meta

    p1 = plt.plot(xp.dfx[region].energy, xp.dfx[region].counts, label=lb)
    cosmetics_plot()
    return p1[0]

def cosmetics_plot(ax = None):
    if ax == None: ax = plt.gca()
    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('CPS [A.U.]')
