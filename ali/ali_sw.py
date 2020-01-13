import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import peakutils
import datetime
import warnings

def load_raw_ali_df ( filename ) :
    """load a df, if filename correctly found, and transform Date/Time from string to Timestamp type"""
    try:
        dfin = pd.read_table(filename, sep='\t', skiprows=1, header=1, decimal=',', encoding='ascii')
        dfin.columns = ["datetime", "p_chamber" ,"p_act", "p_lock",	"p_pre-inj", "t_valve",	"valve", "time_on"]

        dfin["datetime"] = dfin["datetime"].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M:%S.%f'))

        return dfin
    except IOError: warnings.warn(f' does not exist: file = {filename} ', UserWarning)

def plot_pressure_curve(df : pd.DataFrame,
                        tit : str,
                        scale : str = 'min',
                        th : float = 0.1,
                        min_sp : int = 50,
                        flag_p : bool = True,
                        ax = None):
    """Plot ALI pressure curves and apply peak and troughs finding algorithm
    Parameters
    ----------
    df : pandas.Dataframe
        Input data
    tit : str
        Plot title.
    scale : str
        x-axis scale. Accepts 'min' (default), 's' and 'ms'
	th : float
		troughs-to-next-peak time-distance threshold
    min_sp : int
        minimum separation between peaks, default 50 entries
    """

    if scale == 'ms':
        x = 0.1
    elif scale == 's':
        x = 100
    else:
        x = 6000
    if ax == None: ax = plt.gca(); fig = plt.gcf()

    ax.plot(df.index.values/x, df.p_chamber.values, '-', label=tit);

    ax.set_xlabel('Time ['+scale+']', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylabel('Chamber pressure [mbar]', fontsize=14)
    ax.legend(loc=0,fontsize=14);

    if flag_p:
        peaks, troughs = peaks_and_troughs(df.p_chamber.values, min_space=min_sp, th=th)

        ax.plot(troughs/x, df.p_chamber[troughs], 'o', markersize=10, label='__nolegend__')
        ax.plot(peaks/x, df.p_chamber[peaks], '*', markersize=10, label='__nolegend__')
        return [peaks, troughs]

    fig.set_figwidth(10)
    fig.set_figheight(8)

def peaks_and_troughs (y : np.array, min_space : int = 100, th = 0.3) -> (np.array, np.array) :
    """ Find peaks and troughs in an array 'y'
        This function uses the peak detection routine from peakutils
        Parameters
        ----------
        y : np.array
            Array to look for extrema
        min_dist : int
            Minimum index distance between maxima
        th : float
            Threshold for maxima ratio wrt absolute maximum
        Returns
        -------
        peaks : np.array
            Array of maxima
        troughs : np.array
            Array of minima *between the peaks* """

    from peakutils import indexes

    if type(y) == pd.Series:
        y = y.values

    peaks = indexes(y, thres=th, min_dist=min_space)
    troughs = []
    for i, p in enumerate(peaks):
        if i == 0:
            a = 0
        else:
            a = peaks[i-1]
        troughs.append(p-np.argmin(y[p:a:-1]))
    return (peaks, np.array(troughs))

def leaked_mass (df : pd.DataFrame,
                peaks : np.array,
                troughs : np.array,
                M : float = 32.041,
                Seff : float = 66) -> [np.array, np.array, np.array] :
    """Integrate area under ALI pressure curves and compute leak rate and leaked mass.
        Parameters
        ----------
        df : pandas.Dataframe
            Input data
        M : float
            solution/gas molarity
        Seff : float
            pumping effective speed. Default value 66 for Ar (carrier gas) @ 1500 Hz
        Returns
        ----------
        delta_m : np.array
            Array of leaked masses per pulse [g]
        q : np.array
            Array of leak rates per pulse [mbar*L/s]
        delta_t : np.array
            Array of trough to peak time interval (ON time) [s]
        error_delta_m : np.array
            Array of systematic errors for delta_m associated to finite logtime
    """

    logtime = (df.datetime[1]-df.datetime[0]).total_seconds() # [10 ms]
    T = df.t_valve[0]+273 # Pulse valve temperature [K]
    R = 83.144598 # Gas constant [L⋅mbar⋅K−1⋅mol−1]
    y = df.p_chamber.values

    q = []
    delta_m = []
    delta_t = []
    error_delta_m = []

    for trough, peak in zip(troughs, peaks):
        q.append(np.sum(y[trough:peak+1]) * Seff)
        delta_t.append((peak-trough+1) * logtime)
        delta_m.append(q[-1] * M * delta_t[-1]/( R * T))
        error_delta_m.append(delta_m[-1] / delta_t[-1] * logtime)

    return [np.array(delta_m), np.array(q), np.array(delta_t), np.array(error_delta_m)]

def plot_leak_distributions(delta_m : np.array,
                            error_delta_m : np.array,
                            delta_t : np.array,
                            label : str):
    """Plot leaked mass and leak duration over pulses
    and Return average and sd of both arrays"""

    avg_m = np.average(delta_m)
    sd_m = np.std(delta_m)

    avg_t = np.average(delta_t)
    sd_t = np.std(delta_t)

    fig, ax = plt.subplots(2, figsize=(10,20))

    ax[0].errorbar(np.arange(len(delta_m)), delta_m*1e3, error_delta_m*1e3, fmt='o', elinewidth=1, capsize=4, capthick=1)# ecolor='k')
    ax[0].set_xlabel("Number of pulses", fontsize=14)
    ax[0].set_ylabel("Leaked mass [mg]", fontsize=14)

    ax[0].axhline(avg_m*1e3, color='g', label = 'On time = '+label+' ms\n%.1e $\pm$ %.1e mg'%(avg_m*1e3,sd_m*1e3))
    ax[0].fill_between(ax[0].get_xlim(), (avg_m-sd_m)*1e3, (avg_m+sd_m)*1e3, color='0.8')

    ax[0].legend(loc='best', fontsize=14)

    error_delta_t = np.array([0.01 for x in range(len(delta_m))])
    ax[1].errorbar(np.arange(len(delta_t)), delta_t, yerr=error_delta_t, fmt='or')
    ax[1].set_xlabel("Number of pulses", fontsize=14)
    ax[1].set_ylabel("Pulse width [s]", fontsize=14)

    ax[1].axhline(avg_t, color='g',
    label = 'On time = '+label+' ms\n%.2f $\pm$ %.2f s'%(avg_t,sd_t))

    ax[1].fill_between(ax[1].get_xlim(), (avg_t-sd_t), (avg_t+sd_t), color='0.9')
    ax[1].legend(loc='best', fontsize=14)

    return([avg_m, sd_m, avg_t, sd_t])
