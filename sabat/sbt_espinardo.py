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

def get_jpeg_dirs(ipath :str)->List:
    """Get the jpeg dirs in ipath"""
    DIRS =[]
    for r, _, _,  in os.walk(ipath):
        ds =r.split('/')
        if ds[-1] == 'jpeg':
            DIRS.append(r)
    return DIRS


def get_live_dirs(path, sample='sample_5ba'):
    ipath = os.path.join(path, sample)
    live = get_jpeg_dirs(ipath)
    L1 = [l.split('/')[-2] for l in live]
    L2 = [l.split('_')[0] for l in L1]
    L3 = [l.split('_')[2].split(',')[0] for l in L1]
    L4 = [l.split('_')[-1] for l in L1]
    L = list(zip(L2,L3,L4))
    KEYS = [l[0]+ '_' + l[1] + '_' + l[2] for l in L]
    DIR = {}
    for i, k in enumerate(KEYS):
        DIR[k] = live[i]
    print(DIR)
    return DIR


def get_files(ipath :str, ftype : str = 'TOM')->List:
    """Organizes the TOM files or LIVE directories in a dictionary"""

    if ftype == 'TOM':  # get TOM files (.xls extension)
        FLS = glob.glob(ipath+"/*.xls", recursive=True)
        isplit = -1
    else:
        FLS = get_jpeg_dirs(ipath)  # Get LIVE dirs
        isplit = -2

    KEYS = []
    for t in FLS:
        names = t.split('/')[isplit]
        keys  = names.split('_')
        if ftype == 'TOM':
            KEYS.append(keys[0] + '_' + keys[-1].split('.')[0])
        else:
            KEYS.append(keys[0] + '_' + keys[-1])
    DIR = {}
    for i, k in enumerate(KEYS):
        DIR[k] = FLS[i]
    return DIR


def get_TOM_files(ipath, ext='xlsx', rec=False, isplit=-1):
    """Organizes the TOM files in a dictionary"""

    def get_names(file, isplit):
        return file.split('/')[isplit].split('.')[0][0:-1]


    FLS = glob.glob(ipath+f"/*.{ext}", recursive=rec)
    KEYS = []
    FILES = []
    for file in FLS:
        names = get_names(file, isplit)
        KEYS.append(names)
        FILES.append(file)

    TOM={}
    for i, name in enumerate(KEYS):
        TOM[name] = FILES[i]
    return TOM

def select_TOM(TOM, key='A2'):
    TSL={}
    for name, value in TOM.items():
        words = name.split('_')
        if key in words:
            TSL[name] = value
    return TSL


def select_set_TOM(TOM, sample='A2', energy='100mW'):
    SD = select_TOM(TOM, key=sample)
    SE = select_TOM(SD, key=energy)
    return collections.OrderedDict(sorted(SE.items()))


def select_df_TOM(TOM, sample='A2', energy='100mW', filter='Alta450nm'):
    sst = select_set_TOM(TOM, sample, energy)
    SE = select_TOM(sst, key=filter)
    file = list(SE.values())[0]
    tom = pd.read_excel(file, header=None)
    return tom


# def get_TOM_files(ipath :str):
#     """Organizes the TOM files in a dictionary"""
#     FLS = glob.glob(ipath+"/*.xls", recursive=True)
#     isplit = -1
#
#     KEYS = []
#     for t in FLS:
#         names = t.split('/')[isplit]
#         keys  = names.split('_')
#         lbl = keys[0]
#         for i in range(1,5):
#             lbl += '_' + keys[i]
#         lbl +=keys[-1].split('.')[0]
#         KEYS.append(lbl)
#     DIR = {}
#     for i, k in enumerate(KEYS):
#         DIR[k] = FLS[i]
#     return DIR

def read_xls_files(filename :str)->DataFrame:
    """The xls files produced by Espinardo  setup are not really xls
    but tab-separated cvs. This function reads the file and
    returns a DataFrame """

    df = pd.read_csv(filename, delimiter='\t').drop('0,000', 1)
    ndf = df.replace(to_replace=r',', value='.', regex=True)
    return ndf.astype(float)


def sort_by_list(sorting_list, list_to_be_sorted):
    """Sort one list in terms of the other"""
    return [x for _,x in sorted(zip(sorting_list,list_to_be_sorted))]


def get_shot(files : List[str])->List[str]:
    """Gets the shot label in the list"""
    SHOT = []
    for f in files:
        name = f.split('/')[-1]
        jshot = name.split('_')[-1]
        shot  = jshot.split('.')[-2]
        SHOT.append(shot)
    return SHOT


def sort_files(files : List[str])->List[str]:
    """Sort the files by shot order"""

    def get_shot_number(SHOT : List[str])->List[int]:
        """Given a SHOT label get shot number"""
        NSHOT=[]
        for shot in SHOT:
            if shot[-2] == 't':
                shot_number = int(shot[-1])
            else:
                shot_number = int(shot[-2]+shot[-1])
            NSHOT.append(shot_number)
        return NSHOT

    SHOT  = get_shot(files)
    NSHOT = get_shot_number(SHOT)
    return sort_by_list(NSHOT,files)


def load_LIVE_images(files : str)->DataFrame:
    """Load jpg LIVE images and translates them into DFs"""

    #IMG =[]
    DF = []
    SHOT  = get_shot(files)
    print(f'Loading files corresponding to shots {SHOT}')
    for f in files:
        im = Image.open(f)
        npi = np.asarray(im)
        df = pd.DataFrame(npi, index=range(npi.shape[0]))
        #IMG.append(npi)
        DF.append(df)
    return DF
    #return IMG, DF

## fitting

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



def sum_spectrum_region(df, imin, imax):
    asum = 0
    for index, row in df.iterrows():
        if index > imin and index < imax:
            asum = asum + row['I']
        elif index > imax:
            break
    return asum


def tom_I_max(TOMS):
    return [tom.mean().max() for tom in TOMS]

def mean_and_std_toms(TOMS):
    return [tom.mean().mean() for tom in TOMS],[tom.T.mean().std() for tom in TOMS]

def tom_I(TOMS):
    return [tom.mean().sum() for tom in TOMS]

def tom_mean_I(TOMS):
    return [tom.mean().mean() for tom in TOMS]


def avg_intensity(DF):
    return [df.T.mean().mean() for df in DF]


def total_intensity(DF):
    return [df.sum().sum() for df in DF]


def it(df):
    return df.sum().sum()


def itot(dfa, dfb):
    return dfa.sum().sum() + dfb.sum().sum()


def volcm3(xcm,ycm,zmu):
    return xcm * ycm * zmu*1E-4


def cm(mmolMgr, mgr):
    return mmolMgr * mgr * 1E-3 * 6E+23


def cmcm3(mmolMgr, mgr, xcm, ycm, zmu):
    return cm(mmolMgr, mgr) / volcm3(xcm,ycm,zmu)


def norm_fbi_to_pva(apva, cpva, ipva, afbi, cfbi, ifbi):
    """ Returns sigma_fbi/sigma_pva"""
    r= (apva/afbi) * (cfbi/cpva) * (ifbi/ipva)**2
    return 1/r

def total_energy(df):
    return it(df)


def peak_energy(df):
    return df.mean().max()


def peak_energy_ratio_df12(df1,df2):
    return  peak_energy(df1) /  peak_energy(df2)


def total_energy_ratio_df12(df1,df2):
    return  total_energy(df1) /  total_energy(df2)

def total_energy_ratio_dfs(df1,dfs):
    return [total_energy(df1) /  total_energy(df2) for df2 in dfs]

def peak_energy_ratio_dfs(df1,dfs):
    return [peak_energy(df1) /  peak_energy(df2) for df2 in dfs]


def peak_fbi(sdf, bdf, sI, bI):
    s = peak_energy(sdf)
    b = peak_energy(bdf)
    f = s - b * (sI/bI)**2
    return f


def total_fbi(sdf, bdf, sI, bI):
    s = total_energy(sdf)
    b = total_energy(bdf)
    f = s - b * (sI/bI)**2
    print(f'signal at {sI} mW = {s}')
    print(f'bkg at {bI} mW = {b}')
    print(f's - b * ({sI}/{bI})**2 = {f}')
    return f
