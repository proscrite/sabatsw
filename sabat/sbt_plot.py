import numpy as np
import pandas as pd
import os, sys

from  invisible_cities.core.system_of_units import *
import matplotlib.pyplot as plt

from pandas import DataFrame, Series
from matplotlib import cm as cmp

def get_profile(df, cut,  xlsx=True):
    """Gets the profile in Z of the sample"""

    if xlsx:
        return df.mean().index.values, df.mean().values
    else:
        prf  = df[df>=cut].fillna(0).mean()
        indx = prf.index.values[1:-1].astype(str)
        Z    = np.char.replace(indx, ',', '.') .astype(float).astype(int)
        ZV   = prf.values[1:-1]
        return Z, ZV


def display_profile(df, cut=0, zrange=(0,200), yrange=(0,200), xlsx=False, figsize=(12,6)):
    Z,ZV = get_profile(df, cut, xlsx)
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


def display_profiles(dfs, cuts, zrange=(0,200), yrange=(0,200), xlsx=False, figsize=(12,6)):

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)
    df = dfs[0]
    Z,ZV = get_profile(df, cuts[0], xlsx)
    plt.xticks(np.arange(min(Z), max(Z)+1, 10.))
    plt.xlim(*zrange)
    plt.ylim(*yrange)
    plt.xlabel('Z (X) pixels')
    plt.ylabel('I (a.u.)')

    for i, df in enumerate(dfs):
        Z,ZV = get_profile(df, cuts[i], xlsx)
        plt.plot(ZV)
    plt.tight_layout()
    plt.show()

def set_fonts(ax, fontsize=20):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)


def plot_TOM(tom, vmin = 0, interpolation='spline36', cmap='viridis', figsize=(18,12)):
    norm = cmp.colors.Normalize(vmax=tom.max().max(), vmin=vmin)
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)
    if interpolation:
        plt.imshow(tom.values.T, norm=norm, interpolation='spline36', cmap=cmap)
    else:
        plt.imshow(tom.values.T, norm=norm,  cmap=cmap)
    plt.xlabel('X scan')
    plt.ylabel('Z scan')
    plt.show()


def plot_TOMS(toms, xw=2, yw=2, vmin=0, interpolation=False, cmap='viridis', figsize=(18,12)):

    fig = plt.figure(figsize=figsize)
    for i, tom in enumerate(toms):
        ax      = fig.add_subplot(xw, yw, i+1)
        norm = cmp.colors.Normalize(vmax=tom.max().max(), vmin=vmin)
        if interpolation:
            plt.imshow(tom.values.T, norm=norm, interpolation='spline36', cmap=cmap)
        else:
            plt.imshow(tom.values.T, norm=norm, cmap=cmap)
    plt.xlabel('X scan')
    plt.ylabel('Z scan')
    plt.tight_layout()
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
        #print(Z, ZV)
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



def plot_LIVE_images(IMG, nx=6, ny=5, figsize=(18,12)):
    fig = plt.figure(figsize=figsize)
    for i, img in enumerate(IMG):
        ax      = fig.add_subplot(nx, ny, i+1)
        imshow(img,cmap=plt.cm.hot)
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


def plot_spectrum(df, title='I vs W', log=False, figsize=(18,12), fontsize=20):

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    set_fonts(ax, fontsize=fontsize)

    if log:
        ax.set_yscale('log')

    plt.plot(df['W'],df['I'], linewidth=3)
    plt.xlabel('Lambda (nm)')
    plt.ylabel('I (a.u.)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spectra(dfs,
                 labels,
                 title='',
                 xlim=(300,750),
                 log=False,
                 figsize=(18,12), linewidth=2, fontsize=20):

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    set_fonts(ax, fontsize=fontsize)
    if log:
        ax.set_yscale('log')

    for l, df in enumerate(dfs):
        #print(df.W.head())
        plt.xlim(*xlim)
        plt.plot(df['W'],df['I'], linewidth=linewidth, label=labels[l])
    plt.xlabel('Lambda (nm)')
    plt.ylabel('I (a.u.)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fbi(fbi, fbiBa, bkg=False, bkgdf=None,
             lblBKG='Silica', lblFbi="FBI", lblFbiBa="FBI-Ba$^{2+}$",
             line1=428, line2=450, hline=0, xlim=(300,700),
             text=False, xtext=0, ytext=100,
             title = False, grid=False, save=False,
             linewidth=4, figsize=(18,12), fontsize=20):


    font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 30,
        }

    plt.rcParams["font.size"     ] = fontsize

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    set_fonts(ax, fontsize=fontsize)

    plt.xlim(*xlim)
    plt.locator_params(axis='y', nbins=4)
    if bkg:
        plt.plot(bkgdf['W'],bkgdf['I'], linewidth=linewidth, color='k', label=lblBKG)
    plt.plot(fbi['W'],fbi['I'], linewidth=linewidth, color='g', label=lblFbi)
    plt.plot(fbiBa['W'],fbiBa['I'], linewidth=linewidth, color='b',label=lblFbiBa)
    if line1:
        plt.axvline(x=line1,linestyle='dashed', color='k',linewidth=2)
    if line2:
        plt.axvline(x=line2,linestyle='dashed', color='k',linewidth=2)
    if hline:
        plt.axhline(y=hline,linestyle='dashed', color='k',linewidth=2)

    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('Intensity (a.u.)')
    if text:
        plt.text(325, 90, 'a)', fontdict=font)
    if title:
        plt.title(title)

    if grid:
        plt.grid()
    plt.tight_layout()
    plt.legend()
    if save:
        plt.savefig(save)
    plt.show()


def plot_fbis(fbi, dfs,
             lblFbi="FBI", lbldfs=("FBI-Ba$^{2+}$",),
             line1=428, line2=450, hline=0, xlim=(300,700),
             linewidth=4, figsize=(18,12), fontsize=20):


    font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 30,
        }

    plt.rcParams["font.size"     ] = fontsize

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    set_fonts(ax, fontsize=fontsize)

    plt.xlim(*xlim)
    plt.plot(fbi['W'],fbi['I'], linewidth=linewidth, color='r', label=lblFbi)
    for i, fbiBa in enumerate(dfs):
        plt.plot(fbiBa['W'],fbiBa['I'], linewidth=linewidth,label=lbldfs[i])
    if line1:
        plt.axvline(x=line1,linestyle='dashed', color='k',linewidth=2)
    if line2:
        plt.axvline(x=line2,linestyle='dashed', color='k',linewidth=2)
    if hline:
        plt.axhline(y=hline,linestyle='dashed', color='k',linewidth=2)

    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('Intensity (a.u.)')

    plt.tight_layout()
    plt.legend()

    plt.show()


def plot_baseline(fbi, FB = False, figsize=(18,12), fontsize=20, linewidth=2, ms=5,
                  yrange=(0,10),xrange=(300,350)):

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    set_fonts(ax, fontsize=fontsize)

    plt.rcParams["font.size"     ] = fontsize
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rc('axes', linewidth=linewidth)

    font = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': fontsize,
            }


    plt.plot(fbi.W, fbi.I,'ro', ms=ms)

    if FB:
        plt.plot(fbi.W, FB(fbi.W), linewidth=2)

    plt.xlabel('I')
    plt.ylabel('W')
    plt.ylim(*yrange)
    plt.xlim(*xrange)
    plt.grid()
    plt.show()


def plot_baselines(dfs, figsize=(18,12), fontsize=20, linewidth=2, ms=5,
                  yrange=(0,10),xrange=(300,350)):

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    set_fonts(ax, fontsize=fontsize)

    plt.rcParams["font.size"     ] = fontsize
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rc('axes', linewidth=linewidth)

    font = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': fontsize,
            }

    for fbi in dfs:
        plt.plot(fbi.W, fbi.I, 'o', ms=ms)

    plt.xlabel('I')
    plt.ylabel('W')
    plt.ylim(*yrange)
    plt.xlim(*xrange)
    plt.grid()
    plt.show()
