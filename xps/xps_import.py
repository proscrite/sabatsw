import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peakutils
import os
import re
from scipy.optimize import curve_fit

from dataclasses import dataclass

def find_groups(path : str):
    """Utility to find number of groups contained in file"""
    groupCount = 0
    with open(path) as infile:
        for i,line in enumerate(infile):
            if '# Group:' in line :
                groupCount += 1
    return groupCount

def xy_region_delimiters(path: str) -> tuple:
    """Retrieve position, name and number of lines of each spectrum in a .xy file"""

    skipRows0 = []
    nrows0 = []
    names = []

    with open(path) as infile:
        for i,line in enumerate(infile):
            if '# ColumnLabels: energy' in line:
                skipRows0.append(i)
            if '# Region:' in line:
                names.append(line[21:-1].replace(' ', '_'))
            if '# Values/Curve:' in line:
                nrows0.append(int(line[21:-1]))
    return (skipRows0, nrows0, names)

def import_xps_df(path: str) -> pd.DataFrame:
    """Join all spectra in an xps .xy file, each region contains a column with energy [in eV] and count values"""

    skipRows0, nrows0, names = xy_region_delimiters(path) # there are len(skipRows0) - 1 regions in the file

    frames = []
    for j, re in enumerate(skipRows0):
        if j < len(skipRows0):
            frames.append(pd.read_table(path, sep='\s+', skiprows=re+2, nrows = nrows0[j], header=None, names=[names[j], 'counts'],
                                        decimal='.', encoding='ascii', engine='python'))

    dfx = pd.concat(frames, axis=1)

    index2 = np.array(['energy', 'counts'])
    mi = pd.MultiIndex.from_product([names, index2], names=['range', 'properties'])
    mi.to_frame()
    dfx.columns = mi

    return dfx

def excitation_energy_metadata(path : str , name : str):
        """Find the excitation energy for a region in XPS '.xy' file
        Parameters:
        path : str
            Absolute path to file to search into
        name : str
            Name of the region with underscores"""

        with open(path) as infile:
            for i, line in enumerate(infile):
                if name.replace('_', ' ') in line:
                    chunk = infile.readlines(800)
    #                 print(chunk)
                    for li in chunk:
                        if '# Excitation Energy: ' in li:
                            hv = li[21:-1]
                            return float(hv)
            print('Region %s not found' %name)

def ke_to_be(dfx : pd.DataFrame, hv : float) -> pd.DataFrame:
    """Transform energy scale from kinetic to binding"""
    names = list(dfx.columns.levels[0])
    dfnew = pd.DataFrame()

    frames = []
    for n in names:    # Loop over regions
        x = dfx[n].energy.dropna().apply(lambda E : hv - E)  # Subtract hv from KE to yield binding energy
        frames.append( pd.DataFrame([x, dfx[n].counts]).T )
    dfnew = pd.concat(frames, axis=1)

    mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])])
    mi.to_frame()
    dfnew.columns = mi
    return dfnew

def check_arrays(dfr) -> bool:
    """Check whether file is in BE or KE scale"""
    x, y = dfr.dropna().energy.values, dfr.dropna().counts.values
    # Next ensure the energy values are *decreasing* in the array,
    if x[0] < x[-1]:
        is_reversed = True
        return is_reversed
    else:
        is_reversed = False
        return is_reversed

@dataclass
class XPS_experiment:
    """XPS dataclass with regions dfx and metadata
    Attrs:
    -------
    dfx : pd.DataFrame
        table containing all regions found in .xy file
    delimiters : tuple
        position, extension and name of each region to correctly import dfx
    name : str = None
        short name to reference the experiment
    label : str = None
        longer description of the experiment (cleaning, preparation conditions...)
    date : str = None
        experiment date as read in the filename
    other_meta : str = None
        other info contained in the filename
    """
    dfx : pd.DataFrame
    delimiters : tuple
    name : str = None
    label : str = None
    date : str = None
    other_meta : str = None


def xps_data_import(path : str, name : str = None, ) -> XPS_experiment:
    """Method to arrange a XPS_experiment data"""
    dfx = import_xps_df(path)
    delimiters = xy_region_delimiters(path)

    if check_arrays(dfx[delimiters[2][0]]):
        hv = excitation_energy_metadata(path, delimiters[2][0])
        dfx = ke_to_be(dfx, hv)

    relpath, filename = os.path.split(path)
    dir_name = os.path.split(relpath)[1]
    da = re.search('\d+_', filename).group(0).replace('/', '').replace('_', '')
    date = re.sub('(\d{4})(\d{2})(\d{2})', r"\1.\2.\3", da, flags=re.DOTALL)
    other_meta = dir_name + filename.replace(da, '')

    return XPS_experiment(dfx, delimiters, name, date, other_meta)
