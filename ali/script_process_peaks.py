import pandas as pd
import numpy as np

import glob
import os
import sys

import datetime
import warnings

from ali_sw import load_raw_ali_df
from ali_peaks import process_dfRaw_peaks, save_processed_peaks, plotAverageProfile
import argparse

######   Main   ######

parser = argparse.ArgumentParser(description='Process some ALI data.')
parser.add_argument('inPath', metavar='inPath', type=str,
                    help='Path to the Raw ALI data')
parser.add_argument('-L', metavar='peakLength', type=int, nargs=1,
                    default = 0,
                    help='Number of time points')

parser.add_argument('-p', dest='plotter', action='store_const',
                    const=plotAverageProfile,
                    help='Plot average profile with CI')
args = parser.parse_args()

print('Processing file: ', args.inPath)
dfp = process_dfRaw_peaks(args.inPath, peakLength = args.L )#args.peakLength)
save_processed_peaks(args.inPath, dfp)

if args.plotter:
    args.plotter(dfp)
    plt.show(block=True)
