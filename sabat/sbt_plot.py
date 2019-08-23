import numpy as np
import pandas as pd
import os, sys

from  invisible_cities.core.system_of_units import *
import matplotlib.pyplot as plt

from pandas import DataFrame, Series

def get_profile(df, xlsx=True):
    """Gets the profile in Z of the sample"""

    if xlsx:
        return df.mean().index.values, df.mean().values
    else:
        prf  = df.mean()
        indx = df.mean().index.values[1:-1].astype(str)
        Z    = np.char.replace(indx, ',', '.') .astype(float).astype(int)
        ZV   = prf.values[1:-1]
        return Z, ZV


def set_fonts(ax, fontsize=20):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)


def plot_TOM(tom, figsize=(18,12)):
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)
    plt.imshow(tom.values.T)
    plt.xlabel('X scan')
    plt.ylabel('Z scan')
    plt.show()


def plot_tom_set(tset, setsize=2, figsize=(18,12)):
    fig = plt.figure(figsize=figsize)
    for i, (key,file) in enumerate(tset.items()):
        tom = pd.read_excel(file, header=None)
        ax      = fig.add_subplot(setsize, setsize, i+1)
        plt.imshow(tom.values.T)
        plt.xlabel('X scan')
        plt.ylabel('Z scan')
        plt.title(key)
    plt.tight_layout()
    plt.show()


def plot_profile_set(tset, yrange, zrange=(0,150), setsize=2, ptype=True, xlsx=True,
                     figsize=(18,12)):
    IP ={}
    fig = plt.figure(figsize=figsize)
    ir = 0
    for i, (key,file) in enumerate(tset.items()):
        tom = pd.read_excel(file, header=None)

        ax      = fig.add_subplot(setsize, setsize, i+1)
        if ptype:
            Z, ZV = get_profile(tom, xlsx)
        else:
             Z, ZV = get_profile(tom.T, xlsx)
        plt.plot(ZV)
        plt.xticks(np.arange(min(Z), max(Z)+1, 10.))
        plt.xlim(*zrange)
        plt.ylim(*yrange[ir])
        plt.xlabel('Z (X) pixels')
        plt.ylabel('I (a.u.)')
        plt.title(key)
        IP[key] = np.sum(ZV)
        ir+=1
    plt.tight_layout()
    plt.show()
    return IP


def display_profile(df, zrange=(0,200), yrange=(0,200), xlsx=False, figsize=(12,6)):
    Z,ZV = get_profile(df, xlsx)
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


def display_profiles_before_after(dfb, dfa, xlsx=False,
                                  zrange=(0,200), yrange=(0,200),
                                  ztrange=(0,200), ytrange=(0,200),
                                  figsize=(12,6)):

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 2, 1)
    Z, Zb = get_profile(dfb, xlsx)
    Z, Za = get_profile(dfa,xlsx)
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


def display_profiles(DFS, zrange=(0,200), yrange=(0,200), nx = 2, ny =2,
                     xlsx=False, figsize=(12,6)):
    fig = plt.figure(figsize=figsize)
    for i, df in enumerate(DFS):
        Z,ZV = get_profile(df, xlsx)
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
        Z,ZV = get_profile(df.T, xlsx)
        ax      = fig.add_subplot(nx, ny, i+1)
        plt.plot(ZV)
        plt.xticks(np.arange(min(Z), max(Z)+1, 10.))
        plt.xlim(*zrange)
        plt.ylim(*yrange)
        plt.xlabel('X (X) pixels')
        plt.ylabel('I (a.u.)')
    plt.tight_layout()
    plt.show()


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

def show_toms(TOMS, nx = 2, ny =2, figsize=(18,12)):

    fig = plt.figure(figsize=figsize)
    for i, tom in enumerate(TOMS):
        ax      = fig.add_subplot(nx, ny, i+1)
        plt.imshow(tom.values.T)

        plt.xlabel('X scan')
        plt.ylabel('Z scan')
    plt.tight_layout()
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
    plt.show()


def plot_total_intensity(DF, imax = 1e+7, figsize=(12,12)):
    I = total_intensity(DF)
    X = np.arange(len(I))
    fig = plt.figure(figsize=figsize)
    plt.errorbar(X,I, yerr=np.sqrt(I), fmt="kp", ms=7, ls='none')
    plt.ylim(0,imax)
    plt.xlabel('shot number')
    plt.ylabel('I (a.u.)')
    plt.show()
