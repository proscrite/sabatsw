import pandas as pd
import numpy as np

import glob
import os

from ali.ali_sw import load_raw_ali_df
from ali.ali_peaks import *

file300 = '/Users/pabloherrero/sabat/ali_data/Ba(ClO4)2_shooting/23112019_baclo42_FBI_silica_36mg_50ms_800mbar_300pulses'
dftest = process_dfRaw_peaks(file300)


def test_thumbnailPulses(dfptest):
    ax = thumbnailPulses(dfptest)

    assert not ax[0,0].get_yticks().size, "y_ticks not empty"
    assert ax[0,0].get_yscale() == 'log', 'Incorrect y scale'
    assert ax.shape[0] * ax.shape[1] >= Npeaks, "Incorrect ax matrix shape"
    assert len(ax[0,0].lines) == 1, "Overplot on diagonal element"

def test_overplot_every_N_pulses(dfptest):
    every = 30
    ax = overplot_every_N_pulses(dfptest, every=every)
    assert len(ax.lines) == int(Npeaks/every) + 1, "Incorrect number of plotted lines"
