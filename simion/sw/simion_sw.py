import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import glob
import os

def simion_file_metadata(path : str) -> tuple:
    """Retrieve position, name and number of lines of each flight in a simion out file"""
    repet = 0
    skipRows0 = []
    nrows0 = []
    names = []

    with open(path) as infile:
        for i,line in enumerate(infile):
            if '"Ion N","Events","TOF","Mass"' in line:
                repet += 1
                skipRows0.append(i)
                names.append(line[:-1].replace(' ', '_').replace('"', '').split(','))

            if ("------ Begin Next Fly'm ------" in line) and (repet > 0) :
                nrows0.append(i - skipRows0[-1] - 2)

    return (skipRows0, nrows0, names)

def arrange_ions(df0 : pd.DataFrame) -> list:
    df0.drop(1, axis = 0, inplace=True) # Second row is defective, drop it
    df0.reset_index(inplace=True, drop=True)

    mi_ions = pd.MultiIndex.from_arrays([df0.Ion_N, df0.TOF], names=('Ion_N', 'TOF')) # Group each fly by ion and Time Of Flight (TOF)
    mi_ions.to_frame()
    df0.index = mi_ions
    df0.drop(['Ion_N', 'TOF'], axis=1, inplace=True)  # Drop the original columns, leave as multi-index

    return df0

def import_simion_df(path : str, n_flights : int = 0) -> pd.DataFrame:
    """Import SIMION output flights.
    Parameters
    -------------
    path : str
        full path to data file. Metadata containing position and length of each flight is automatically read
    n_flights : int = 0
        number of flights to import in file. If unspecified, will import all of them

    Returns
    -------------
    List of dataframes for each flight, neatly arranged by ion number
    """
    frames = []
    skipRows0, nrows0, names = simion_file_metadata(path)

    if n_flights > len(skipRows0):
        print('File only contains %i flights, will exit' %len(skipRows0))
        return
    if n_flights == 0:      # If not specified, import all flights
        n_flights = len(skipRows0)

    for j, re in enumerate(skipRows0 [: n_flights] ):
        if j < len(skipRows0)-1:
            # Import fly by fly (until nrows0)
            df0 = pd.read_csv(path, skiprows = re, header = 0, nrows = nrows0[j], names=names0[j], engine='python')
            df1 = arrange_ions(df0)
            frames.append(df1)
        else:
            # For last fly import all remaining rows
            df0 = pd.read_csv(path, skiprows = re, header = 0, names=names0[j], engine='python')
            df1 = arrange_ions(df0)
            frames.append(df1)

    return frames
