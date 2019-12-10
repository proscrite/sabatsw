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


def get_TOM_files(ipath :str):
    """Organizes the TOM files in a dictionary"""
    FLS = glob.glob(ipath+"/*.xls", recursive=True)
    isplit = -1

    KEYS = []
    for t in FLS:
        names = t.split('/')[isplit]
        keys  = names.split('_')
        lbl = keys[0]
        for i in range(1,5):
            lbl += '_' + keys[i]
        lbl +=keys[-1].split('.')[0]
        KEYS.append(lbl)
    DIR = {}
    for i, k in enumerate(KEYS):
        DIR[k] = FLS[i]
    return DIR

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


# plots

def get_profile(df):
    """Gets the profile in Z of the sample"""
    prf  = df.mean()
    indx = df.mean().index.values[1:-1].astype(str)
    Z    = np.char.replace(indx, ',', '.') .astype(float).astype(int)
    ZV   = prf.values[1:-1]
    return Z, ZV

def display_profile(df, zrange=(0,200), yrange=(0,200), figsize=(12,6)):
    Z,ZV = get_profile(df)
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)
    plt.plot(ZV)
    plt.xticks(np.arange(min(Z), max(Z)+1, 10.))
    plt.xlim(*zrange)
    plt.ylim(*yrange)
    plt.xlabel('Z (X) pixels')
    plt.ylabel('I (a.u.)')
    plt.tight_layout()
    plt.show()


def display_profiles_before_after(dfb, dfa,
                                  zrange=(0,200), yrange=(0,200),
                                  ztrange=(0,200), ytrange=(0,200),
                                  figsize=(12,6)):

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 2, 1)
    Z, Zb = get_profile(dfb)
    Z, Za = get_profile(dfa)
    plt.plot(Zb)
    plt.plot(Za)
    plt.xlim(*zrange)
    plt.ylim(*yrange)
    plt.xticks(np.arange(min(Z), max(Z)+1, 10.))
    plt.xlabel('Z (X) pixels')
    plt.ylabel('I (a.u.)')

    ax      = fig.add_subplot(1, 2, 2)
    Z,Zb = get_profile(dfb.T)
    Z,Za = get_profile(dfa.T)
    plt.plot(Zb)
    plt.plot(Za)
    plt.xlim(*ztrange)
    plt.ylim(*ytrange)
    plt.xticks(np.arange(min(Z), max(Z)+1, 10.))
    plt.xlabel('Z (X) pixels')
    plt.ylabel('I (a.u.)')

    plt.tight_layout()
    plt.show()

def display_profiles(DFS, zrange=(0,200), yrange=(0,200), nx = 2, ny =2, figsize=(12,6)):
    fig = plt.figure(figsize=figsize)
    for i, df in enumerate(DFS):
        Z,ZV = get_profile(df)
        ax      = fig.add_subplot(nx, ny, i+1)
        plt.plot(ZV)
        plt.xticks(np.arange(min(Z), max(Z)+1, 10.))
        plt.xlim(*zrange)
        plt.ylim(*yrange)
        plt.xlabel('Z (X) pixels')
        plt.ylabel('I (a.u.)')
    plt.tight_layout()
    plt.show()
    fig = plt.figure(figsize=figsize)
    for i, df in enumerate(DFS):
        Z,ZV = get_profile(df.T)
        ax      = fig.add_subplot(nx, ny, i+1)
        plt.plot(ZV)
        plt.xticks(np.arange(min(Z), max(Z)+1, 10.))
        plt.xlim(*zrange)
        plt.ylim(*yrange)
        plt.xlabel('X (X) pixels')
        plt.ylabel('I (a.u.)')
    plt.tight_layout()
    plt.show()

def show_toms(TOMS, nx = 2, ny =2, figsize=(18,12)):

    fig = plt.figure(figsize=figsize)
    for i, tom in enumerate(TOMS):
        ax      = fig.add_subplot(nx, ny, i+1)
        plt.imshow(tom.values.T)

        plt.xlabel('X scan')
        plt.ylabel('Z scan')
    plt.tight_layout()
    plt.show()

def mean_and_std_toms(TOMS):
    return [tom.mean().mean() for tom in TOMS], [tom.mean().std() for tom in TOMS]

def tom_I(TOMS):
    return [tom.mean().sum() for tom in TOMS]


def tom_mean_I(TOMS):
    return [tom.mean().mean() for tom in TOMS]


def plot_LIVE_images(IMG, nx=6, ny=5, figsize=(18,12)):
    fig = plt.figure(figsize=figsize)
    for i, img in enumerate(IMG):
        ax      = fig.add_subplot(nx, ny, i+1)
        imshow(img,cmap=plt.cm.hot)
    plt.show()


def plot_LIVE_avg(DF, xpixel =(0, 255), ypixel = (0,255), imax = 200, nx=6, ny=5, figsize=(18,12)):
    xmin = xpixel[0]
    xmax = xpixel[1]
    ymin = ypixel[0]
    ymax = ypixel[1]
    def plot_df(DF, axis='X'):
        fig = plt.figure(figsize=figsize)
        for i, img in enumerate(DF):
            ax      = fig.add_subplot(nx, ny, i+1)
            if axis == 'X' :
                plt.plot(DF[i].mean()[xmin:xmax])
            else:
                plt.plot(DF[i].T.mean()[ymin:ymax])
            plt.xlabel(axis + ' (pixel)')
            plt.ylabel('I (a.u.)')
            plt.ylim(0,imax)
        plt.show()
    plot_df(DF, axis='X')
    plot_df(DF, axis='Y')


def plot_avg_intensity(DF, imax = 200, err=None, figsize=(12,12)):
    I = avg_intensity(DF)
    X = np.arange(len(I))
    fig = plt.figure(figsize=figsize)
    if err == None:
        err = np.sqrt(I)
    plt.errorbar(X,I, yerr=err, fmt="kp", ms=7, ls='none')
    plt.ylim(0,imax)
    plt.xlabel('shot number')
    plt.ylabel('I (a.u.)')
    #plt.show()

def plot_total_intensity(DF, imax = 1e+7, figsize=(12,12)):
    I = total_intensity(DF)
    X = np.arange(len(I))
    fig = plt.figure(figsize=figsize)
    plt.errorbar(X,I, yerr=np.sqrt(I), fmt="kp", ms=7, ls='none')
    plt.ylim(0,imax)
    plt.xlabel('shot number')
    plt.ylabel('I (a.u.)')
    #plt.show()

def avg_intensity(DF):
    return [df.T.mean().mean() for df in DF]


def total_intensity(DF):
    return [df.sum().sum() for df in DF]


def plot_TOM(tom, figsize=(18,12)):
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)
    plt.imshow(tom.values.T)
    plt.xlabel('X scan')
    plt.ylabel('Z scan')
    plt.show()
