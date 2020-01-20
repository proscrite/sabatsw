import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from scipy.optimize import curve_fit

from dataclasses import dataclass
from xps.xps_import import *
from xps.xps_analysis import *

def test_compute_p0_peaks():
    sample_file = '/Users/pabloherrero/sabat/xps_spectra/2019_10_28_FBI_dropcast_Au_crystal/20191028_FBI_dropcast_1mM_Au(788).xy'
    xp_double = xps_data_import(sample_file)
    x, y = xp_bg.dfx.Ba_3d.dropna().energy.values, xp_bg.dfx.Ba_3d.dropna().counts.values
    p0 = compute_p0_peaks(x, y, thres0 = 0.5, Npeaks=2)
    assert len(p0) == 6, "The sample data should yield 2 peaks, i.e. 6 fit params."

def test_fit_double_gauss():
    sample_file = '/Users/pabloherrero/sabat/xps_spectra/2019_10_28_FBI_dropcast_Au_crystal/20191028_FBI_dropcast_1mM_Au(788).xy'
    xp_double = xps_data_import(sample_file)
    xp_smooth = gaussian_smooth(xp_double, 'Ba_3d');
    xp_bg = subtract_double_shirley(xp_smooth, 'Ba_3d', 790);
    fitp = fit_double_gauss(xp_bg, 'Ba_3d', thres0=0.5)
    plt.clf();
    assert int(fitp[0]) == 797, "Incorrect position of the Ba 3d 3/2 peak "
    assert int(fitp[1]) == 782, "Incorrect position of the Ba 3d 5/2 peak "
