from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import glob
import os

from typing      import Tuple
from typing      import Dict
from typing      import List
from typing      import TypeVar
from typing      import Optional

from enum        import Enum

from dataclasses import dataclass
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import invisible_cities.core .fit_functions  as     fitf
from collections import Counter
import collections

## File manipulation

def error_ratio(a, b, sa, sb):
    sx2 = (1 / b**2) * (sa**2 + (a/b)**2 * sb**2)
    return np.sqrt(sx2)

def error_prod(a, b, sa, sb):
    sx2 = b**2 * sa**2 + a**2 * sb**2
    return np.sqrt(sx2)

def expo_seed(x, y, eps=1e-12):
    """
    Estimate the seed for a exponential fit to the input data.
    """
    x, y  = zip(*sorted(zip(x, y)))
    const = y[0]
    slope = (x[-1] - x[0]) / np.log(y[-1] / (y[0] + eps))
    seed  = const, slope
    return seed


def fit_intensity(DF, sigma, imax=200, figsize=(10,10)):
    I = avg_intensity(DF)
    X = np.arange(len(I))
    seed = expo_seed(X, I)
    f    = fitf.fit(fitf.expo, X, I, seed, sigma= sigma * np.ones(len(I)))

    fig = plt.figure(figsize=figsize)
    plt.errorbar(X,I, fmt="kp", yerr= sigma  * np.ones(len(I)), ms=7, ls='none')
    plt.plot(X, f.fn(X), lw=3)
    plt.ylim(0,imax)
    plt.xlabel('shot number')
    plt.ylabel('I (a.u.)')
    plt.show()
    print(f'Fit function -->{f}')
    return f.values, f.errors


def get_columns_from_names(series_name='S9A', wavelength_nm='250'):
    title = f"{series_name}_{wavelength_nm}"
    w = f"{title}_W"
    i = f"{title}_I"
    return i, w


def get_WI_from_df(df, I, W='W', scale=1):
    w  = df[W]
    i  = df[I] * scale
    si = np.sqrt(abs(i))

    return pd.concat([w,i,si], axis=1, keys=['W','I','SI'])


def get_spectrum_from_df(df, series_name='S9A', wavelength_nm='250'):
    I,W = get_columns_from_names(series_name, wavelength_nm)
    w  = df[W]
    i  = df[I]
    si = np.sqrt(i)

    return pd.concat([w,i,si], axis=1, keys=['W','I','SI'])

def get_spectrum_from_pills(df, series_name='S9A'):
    w  = df['W']
    i  = df[series_name]
    si = np.sqrt(i)

    return pd.concat([w,i,si], axis=1, keys=['W','I','SI'])


def subtract_spectra2(df1, df2, df3):
    df12 = subtract_spectra(df1, df2)
    return subtract_spectra(df12, df3)


def subtract_spectra(df1, df2):
    def subtract_series(df1, df2):
        i1 = df1['I']
        i2 = df2['I']
        si1 = df1['SI']
        si2 = df2['SI']
        w  = df1['W']
        i  = i1 - i2
        si = np.sqrt(si1 + si2)
        return i, w, si

    i, w, si = subtract_series(df1, df2)
    return pd.concat([w,i,si], axis=1, keys=['W','I','SI'])



def add_spectra(df1, df2, w1, w2):
    def add_series(df1, df2, w1, w2):
        i1 = df1['I']
        i2 = df2['I']
        si1 = df1['SI']
        si2 = df2['SI']
        w  = df1['W']
        i  = w1 * i1 +  w2 * i2
        si = np.sqrt(si1 + si2)
        return i, w, si

    i, w, si = add_series(df1, df2, w1, w2)
    return pd.concat([w,i,si], axis=1, keys=['W','I','SI'])

def subtract_spectra_thr_zero(df1, df2):
    df = subtract_spectra(df1, df2)
    for index, row in df.iterrows():
        if row['I'] < 0:
            df.loc[index,'I'] = 0
    return df


def subtract_spectra_thr_pzero(df1, df2):
    df = subtract_spectra(df1, df2)
    for index, row in df.iterrows():
        if row['I'] < 1:
            df.loc[index,'I'] = 1
    return df


def subtract_spectra_thr_bkg(df1, df2):
    df = subtract_spectra(df1, df2)
    for index, row in df.iterrows():
        if row['I'] < row['SI']:
            df.loc[index,'I'] = row['SI']
    return df


def scale_spectra(df, scale=1000):
    def scale_series(df):
        i  = df['I'] * scale
        si = df['SI'] * np.sqrt(scale)
        w  = df['W']
        return i, w, si

    i, w, si = scale_series(df)
    return pd.concat([w,i,si], axis=1, keys=['W','I','SI'])


def subtract_series(df1, df2, name1, name2, wavelength_nm='250'):
    i1 = df1[f'{name1}_{wavelength_nm}_I']
    i2 = df2[f'{name2}_{wavelength_nm}_I']
    w  = df1[f'{name1}_{wavelength_nm}_W']
    i  = i1 - i2
    si = np.sqrt(i1 + i2)
    return i,w,si


def loc_w(df, wi, tol = 0.2):
    w  = df['W']
    return w[abs(w-wi) < tol].index[0]



def sum_spectrum_region(df, imin, imax):
    asum = 0
    for index, row in df.iterrows():
        if index > imin and index < imax:
            asum = asum + row['I']
        elif index > imax:
            break
    return asum

def sum_spectrum_error_region(df, imin, imax):
    asum = 0
    for index, row in df.iterrows():
        if index > imin and index < imax:
            asum = asum + row['SI'] * row['SI']
        elif index > imax:
            break
    return np.sqrt(asum)


def interval_area(df, wi, wc):
    i0  = loc_w(df, wi)
    ic  = loc_w(df, wc)

    return sum_spectrum_region(df, i0, ic)


def interval_error_area(df, wi, wc):
    i0  = loc_w(df, wi)
    ic  = loc_w(df, wc)

    return sum_spectrum_region(df, i0, ic), sum_spectrum_error_region(df, i0, ic)


def volcm3(xcm,ycm,zmu):
    return xcm * ycm * zmu*1E-4


def n_m(mmolMgr, mgr):
    return mmolMgr * mgr * 1E-3 * 6E+23


def cmcm3(mmolMgr, mgr, xcm, ycm, zmu):
    return n_m(mmolMgr, mgr) / volcm3(xcm,ycm,zmu)


def norm_fbi_to_pva(apva, cpva, ipva, afbi, cfbi, ifbi):
    """ Returns sigma_fbi/sigma_pva"""
    r= (apva/afbi) * (cfbi/cpva) * (ifbi/ipva)**2
    return 1/r


def area(df, wi, wc):
    i0  = loc_w(df, wi)
    ic  = loc_w(df, wc)

    return sum_spectrum_region(df, i0, ic)


def disc_factor(fbi, fbiBa, wi, wl, wc):
    def frac_area(df, i0, il, ic):
        df_total_area = sum_spectrum_region(df, i0, il)
        df_c          = sum_spectrum_region(df, i0, ic)
        return df_c, df_total_area, df_c / df_total_area

    i0  = loc_w(fbi, wi)
    il  = loc_w(fbi, wl)
    ic  = loc_w(fbi, wc)
    fbi_c, fbi_t, fbi_f       = frac_area(fbi,   i0, il, ic)

    i0  = loc_w(fbiBa, wi)
    il  = loc_w(fbiBa, wl)
    ic  = loc_w(fbiBa, wc)
    fbiBa_c, fbiBa_t, fbiBa_f = frac_area(fbiBa, i0, il, ic)

    r_f = fbiBa_f / fbi_f

    return r_f, fbi_c, fbi_t, fbi_f, fbiBa_c, fbiBa_t, fbiBa_f


def disc_factor_and_error(fbi, fbiBa, wi, wl, wc):

    r_f, fbi_c, fbi_t, fbi_f, fbiBa_c, fbiBa_t, fbiBa_f = disc_factor(fbi, fbiBa, wi, wl, wc)

    efbi_f   = fbi_f * np.sqrt((1/fbi_c) + (1/fbi_t))
    efbiBa_f = fbi_f * np.sqrt((1/fbiBa_c) + (1/fbiBa_t))

    r_f = fbiBa_f / fbi_f
    er_f = r_f * np.sqrt((efbi_f/fbi_f)**2 + (efbiBa_f/fbiBa_f)**2)

    return fbi_f, efbi_f, fbiBa_f, efbiBa_f, r_f, er_f


def rdf(fbi, fbiBa, wi, wl, wmin, wmax):
    R = [disc_factor(fbi, fbiBa, wi, wl, wc) for wc in range(wmin, wmax,2)]
    r_f, *_ = zip(*R)
    return [wc for wc in range(wmin, wmax,2)], r_f
