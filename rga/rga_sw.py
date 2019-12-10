#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import glob
import os
import peakutils
import datetime
import warnings

# #####  Import structure:
#  - Import one spectrum with corresponding spectrum number (0-n_spectra) and datetime
#  - Infer nrows for each spectrum
#  - Infer number of spectra
#  - Import recursively next spectra with datetime and discard mass column (common)
#  - Concatenate columns in final dataframe
#  - Infer mass step from total range (200 amu) and first step
######

from dataclasses import dataclass

@dataclass
class ProcessScanDF:
    """Method class Process Raw Scan DataFrame"""
    filename : str
    test : bool = False
    Nrows : int = 0
    skipRows0 : int = 0
    buffer_lines : int = 0
    n_spectra : int = 0

    def retrieve_metadata(self, head_size: int = 2000):
        """Compute from file occurrences the following parameters:
            - skiprows0: number of header lines before first spectrum data
            - Nrows : number of rows each spectrum occupies
            - buffer_lines : number of buffer lines between one spectrum and the next
            - n_spectra : total number of spectra found
        """
        of = open(self.filename)
        head = [next(of) for x in range(head_size)]
        repet = 0

        for i,line in enumerate(head):
            if 'Mass [amu]' in line:
                repet += 1
                if repet == 1:
                    self.skipRows0 = i
    #                 print('skipRows0 = ', skipRows0)
                elif repet == 2:
                    self.buffer_lines = i - self.Nrows - self.skipRows0
    #                 print('buffer_lines = ', buffer_lines)

            if 'Task Name' in line and repet == 1:
                self.Nrows = i - self.skipRows0 - 2

            number_lines = sum(1 for line in open(self.filename))
        if self.n_spectra == 0:
            self.n_spectra = round(number_lines/(self.Nrows+self.buffer_lines))
            print(self.n_spectra)

        def retrieve_advanced_metadata(head : list, skipRows0 : int, Nrows: int):
            """Compute first mass, scan width (max. 200 amu) and resolution (fraction of amu read) from file"""
            m0 = float(str.rsplit(head[self.skipRows0 + 1], sep='\t')[0].replace(',','.'))
            m1 = float(str.rsplit(head[self.skipRows0 + 2], sep='\t')[0].replace(',','.'))
            resolution = m1 - m0

            first_mass = float(str.rsplit(head[self.Nrows + self.skipRows0 + 3])[-1].replace(',', '.'))
            scan_width = float(str.rsplit(head[self.Nrows + self.skipRows0 + 4])[-1].replace(',', '.'))

            return(resolution, first_mass, scan_width)

#         return [Nrows, skipRows0, buffer_lines]

    def import_first_column(self)->pd.DataFrame:
        """Import first spectrum to stitch to the rest and keep 'mass [amu]' column"""

        df0 = pd.read_csv(self.filename, sep='\t', skiprows= self.skipRows0 - 2, nrows = self.Nrows,
                      header=1, decimal=",", names=['mass', 'ion_current'], encoding='ascii')
        return df0

    def import_main_no_test(self) -> pd.DataFrame:
        """Main dataframe importing function. Read n_spectra using metadata obtained previously
        %%timeit result: 259 ms ± 4.59 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        """

        df1 = pd.concat([pd.read_csv(self.filename, sep='\t', skiprows = i * ( self.Nrows + self.buffer_lines ) + self.buffer_lines + 2,
                                     nrows = self.Nrows, header=1, decimal=',', encoding='ascii',
                                     names=['a', 'ion_current']).drop('a', axis=1) for i in range(1, self.n_spectra)], axis=1)
        return df1

    def import_main_test(filename : str, buffer_lines: int, Nrows : int, n_spectra : int) -> pd.DataFrame:
        """Main dataframe importing function tested for first_mass and scan_range matching
        %%timeit result 254 ms ± 18.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        """

        df1 = pd.concat([pd.read_csv(self.filename, sep='\t', skiprows = i * ( self.Nrows + self.buffer_lines ) + self.buffer_lines + 2,
                                     nrows = self.Nrows, header=1, decimal=',', encoding='ascii',
                                     names=['a', 'ion_current']) for i in range(1, self.n_spectra)], axis=1)

        assert (df1.a.iloc[0] == first_mass).all(), "The read spectra do not match the expected first mass"
        assert (round(df1.a.iloc[-1]) == scan_width).all(), "The read spectra do not match the expected width"

        df1 = df1.drop(columns='a')
        return df1

    def read_start_datetimes(self) -> pd.DataFrame:
        """Read Start Time of each cycle and convert to datetime"""

        dtstr = np.array([pd.read_csv(self.filename, sep='\t', skiprows= i * (self.Nrows + self.buffer_lines) + self.buffer_lines + 2,
                                      nrows=1, header=None).iloc[0,1] for i in range(self.n_spectra)])
        print(dtstr)
        try:
            dt = np.array([datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S.%f ') for x in dtstr])
        except ValueError:
            dt = np.array([datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S.%f') for x in dtstr])
        return dt

    def export_df(self, df : pd.DataFrame):
        """Write stitched dataframe to '.txt' file with same basename"""

        outPath = os.path.dirname(self.filename)
        outName = os.path.splitext(os.path.basename(self.filename))[0]
        outFile = os.path.join(outPath, outName+'.txt')

        df.to_csv(outFile)

    def main(self):
        """Call of all functions, stitch dataframes and export to '.txt'
            - df0 : first cycle including mass column
            - df1 : rest of the cycles with mass column dropped
            - dt : datetimes to use as column names"""
#         [self.Nrows, self.skipRows0, self.buffer_lines] = self.retrieve_metadata()
        self.retrieve_metadata()
        df0 = self.import_first_column()

        ##
        #   number_lines = sum(1 for line in open(filename))
        #
        #if n_spectra == None:
        #n_spectra = round(number_lines/Nrows)
        ##

        if self.test:
            df1 = self.import_main_test()
        else:
            df1 = self.import_main_no_test()

        dt = self.read_start_datetimes()

        df = pd.concat([df0,df1], axis=1)
        df.set_index(['mass'], inplace=True)
        df.columns = [dt]

        self.export_df(df)

def process_raw_rga_data(filename : str, n_spectra : int = None,
                spectrum_range : int = 200, resolution : int = 32, buffer_lines : int = 8):
    """Load rga spectra in file 'filename'
    Parameters
    ----------
    filename : path
        Absolute file path, extension '.asc'
    n_spectra : int
        Number of spectra to load. If 'None' passed, compute total number of spectra in file

    spectrum_range : float = 200
        Length of spectrum in amu
    resolution : int = 32
        Points per amu
    buffer_lines : int = 8
        number of lines between spectra
    """

    number_lines = sum(1 for line in open(filename))

    Nrows = spectrum_range * resolution
    if n_spectra == None:
        n_spectra = round(number_lines/Nrows)

    df0 = pd.read_csv(filename, sep='\t', skiprows=buffer_lines+3, nrows=Nrows-buffer_lines, header=1, decimal=",", names=['mass', 'ion_current'], encoding='ascii')
    # df0['mass'] = [x.replace(',', '.') for x in df0['mass']]
    # df0.mass.astype(np.float64, decimal=',')

    # # Import next spectrum

    # ## Concat all spectra dropping the 'mass' column

    df1 = pd.concat([pd.read_csv(filename, sep='\t', skiprows= i * Nrows + buffer_lines+2, nrows=Nrows-buffer_lines,
                          header=1, decimal=',', encoding='ascii', names=['a', 'ion_current']).drop('a', axis=1) for i in range(1, n_spectra)], axis=1)

    # ## Find all datetimes

    dtstr = np.array([pd.read_csv(filename, sep='\t', skiprows=Nrows * i + buffer_lines+2, nrows=1, header=None).iloc[0,1] for i in range(n_spectra)])

    try:
        dt = np.array([datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S.%f ') for x in dtstr])
    except ValueError:
        dt = np.array([datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S.%f') for x in dtstr])

    # ## Append first spectrum and set column names from dt

    df = pd.concat([df0,df1], axis=1)
    df.set_index(['mass'], inplace=True)
    df.columns = [dt]

    outPath = os.path.dirname(filename)
    outName = os.path.splitext(os.path.basename(filename))[0]
    outFile = os.path.join(outPath, outName+'.txt')

    df.to_csv(outFile)

    return df


def compress_df(df : pd.DataFrame, xrange : int = 200, resolution : int = 32) -> pd.DataFrame:
    """Compress raw rga signal dataframe from resolution to 1 value per bin"""
    dt = df.columns.values
    #dt = [ i[0] for i in df.columns.values]

    mat = [df.iloc[i * resolution : (i+1) * resolution,:].mean(axis=0) for i in range(xrange)]
    df2 = pd.DataFrame(np.array(mat))
    df2.columns = dt
    return df2

def load_processed_rga_data(filename: str) -> pd.DataFrame:
    """Load pre-processed rga '.txt' data. Output from process_raw_rga_data.
    Arguments
    ----------
    filename : str
        Relative path and file name with extension '.txt'
    Returns
    ----------
    dfp : pd.DataFrame
        rga processed df with every cycle stored on a column
    """

    dfp = pd.read_csv(filename)

    dfp.iloc[:,0]
    dfp.set_index(dfp.iloc[:,0], drop=True, inplace=True, verify_integrity=True)
    dfp.drop(dfp.index[0], inplace=True)
    dfp.index.name = 'mass'
    dfp.drop('Unnamed: 0', axis=1, inplace=True)

    return dfp

def find_excess_peaks(df : pd.DataFrame, i_cycle : int, f_cycle : int,
                      mass_start : int = 0, mass_end : int = 200,
                     th : float = 0.2, p_flag : bool = True, figsize : tuple = (10,8)):

    """Compute and plot residual peaks with respect to average in the passed df itself. The df must be compressed"""

    av_sign = df.mean(axis=1)
    plt.figure(figsize=(figsize))
    offset = min(av_sign)

    avY = np.zeros_like(av_sign[mass_start : mass_end])

    for i in range(i_cycle,f_cycle):
        y = (df.iloc[mass_start : mass_end,i] - av_sign[mass_start : mass_end])
        avY += y
    #     plt.bar(compDf.index[130*:], y, '.-', linewidth=0.1, label=lb)

    avY /= f_cycle-i_cycle
    plt.bar(avY.index, avY.values, color='b', label='signal-bg')

    plt.bar(av_sign[mass_start : mass_end].index, av_sign[mass_start : mass_end].values , color='r', linewidth=2, label='Bg: Whole average signal')
    # plt.fill_between((av_sign[mass_start : mass_end] - sd_comp[mass_start : mass_end]) / offset, (av_sign[mass_start : mass_end] + sd_comp[mass_start : mass_end]) / offset, color='0.6')
    if p_flag:
        ind = peakutils.indexes(avY, thres=th)
        ind = [round(i,2) + mass_start for i in ind]
        plt.plot(ind, avY[ind], 'og', markersize=10, label='Peaks at threshold %.2f'%th)
        for i, j in enumerate(ind):
            plt.text(j, avY[j]*1.1, str(j), bbox=dict(fc='w'), fontsize = 14)

    plt.xlabel('mass [amu]', fontsize=14)
    plt.ylabel('spectrum residuals [A]', fontsize=14)
    plt.title('Evaporation run. Residuals of cycles %i-%i '%(i_cycle,f_cycle), fontsize=16)
    plt.yscale('log')

    plt.legend(loc='upper right', fontsize=14)

    plt.show()


def load_mid_data(filename : str) -> pd.DataFrame:
    """Load data from filename_MID_.asc into DataFrame df_mid with MultiIndex:
        Level 0 are the ion_masses as they appear in the file line 7
        Level 1 are the (timestamp, relative time and ion_current [A]) for each ion"""

    head = pd.read_csv(filename, sep='\s+', skiprows=5, nrows=1)
    names = head.columns.values
    nions = len(names)
    
    ##### Establish the column names (timestamp, relative_time and ion_current) as index in level 1
    index2 = np.array(['time', 'rel_time', 'ion_current'])

    #####  Declare a MultiIndex object from the euclidean product ion_masses $\times$ properties
    mi = pd.MultiIndex.from_product([names, index2], names=['mass', 'properties'])

    ##### Read the actual data skipping headers and set the MultiIndex to its columns
    myparser = lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S.%f ')
    df_mid = pd.read_csv(filename, sep='\t', skiprows=8, header=None, decimal=',',
                        names=[str(x) for x in range(3*nions)],
                        parse_dates=[str(x) for x in range(0,nions,3)], date_parser=myparser)
    df_mid.columns = mi

    return df_mid

def plot_MID(df_mid : pd.DataFrame, masses : list, xlims : tuple):
    """Plot Multi Ion Data from RGA df_mid (rel_time vs ion_current)
    Parameters
    ---------------
    df_mid : pd.DataFrame
        MID-type RGA data
    masses: list
        ion masses to plot as they appear in df_mid
    xlims: tuple
        limits on x axis (in seconds)

    Hint: to access all the names of the ions from df_ion use
        df.columns.get_level_values(level=0).drop_duplicates()
    """

    plt.figure(figsize=(10,8))
    for m in masses:
        x = df_mid[m].rel_time
        y = df_mid[m].ion_current

        plt.plot(x, y, '-', label='Mass peak: '+m)

    if xlims != None:
        plt.xlim(xlims)
    plt.xlabel('Relative time [s]')
    plt.ylabel('Ion current [A]')
    plt.yscale('log')
    plt.legend()
