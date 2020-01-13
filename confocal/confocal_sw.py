import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import os
import sys
import glob
import re

from dataclasses import dataclass

def load_jpg_path(files : str) -> [str, str] :
    """Load confocal .jpg paths and separate in fluorescence (f1, ch00) and topology paths (f2, ch01)
    -------
    Default file path : ~/Documents/confocal/
    --------
    Parameters:
    files : str
        path of experiment directory"""

    dir_def = f"/Users/pabloherrero/Documents/confocal/"
    f = dir_def + files + '/Series*.jpg'
    print(f)
    f = glob.glob(os.path.expandvars(f))
    f.sort()
    f1, f2 = chaffFromWheat(f)
    print("Found ", len(f1), len(f2), " files")
    return [f1, f2]

def findSetEnd(s : str, f : list, verbose : bool = False):
    f.sort()
    res = []
    for i,p in enumerate(f):
        if s in p:
            if verbose:
                print(i,p)
            res.append(i)
    return res

def plot_both_channels(f: list, end: int = -1, size: float = 10):
    """ Plot fluorescence and topography channel side by side
    f : list -> list of image files to plot
    end : int -> (optional) range limit of files to plot from f
    size : float -> image size"""
    nrow = int(end/2)
    fig, axarr = plt.subplots(nrow,2, sharex='all', sharey='all', figsize=(size,nrow/2*size))
    for index, im in enumerate(f[:end]):
        i, j = int(index/2), index%2
        a = plt.imread(im)
        axarr[i, j].imshow(a)
        axarr[i, j].axis('off')
    # plt.tight_layout(pad=0, h_pad=0)
    plt.subplots_adjust(left=0.001, right=0.999, top=0.9, bottom=0.001, hspace=0.1, wspace=0.1)

def chaffFromWheat(f : list):
    """Separate fluorescence files ('ch00') from topography ones ('ch01')
    Returns two lists w/ each of them"""
    f.sort()
    topog = []
    fluor = []
    for p in f:
        if 'ch00' in p:
            # Files w/ 'ch00' are scan 1: blue light, fluorescence. Scan 2 ('ch01') is topography by reflection of green
            fluor.append(p)
        else:
            topog.append(p)
    return [fluor, topog]

def plot_whole_grid(f: list, cols: int = None, rows: int = None,
                    size: float = 1,
                   rightToLeft = False, vertPlot = True,
                   lmargin=0.01, gap=0.01,
                   title=False):
    """ Plot whole grid of one channel of images
    f : list -> list of image files to plot
    cols : int -> number of columns in the grid to plot
    rows : int -> number of rows in the grid [set to one for 1D axarr]
    size : float -> image size
    rightToLeft : bool -> flag to plot columns from right to left
    vertPlot : bool -> flag to plot from up to down. If need to plot only one column use this flag
    title : bool -> display image title"""

    if cols == None:
        cols = len(findSetEnd('t00', f))
        if cols == 0:
            cols = len(findSetEnd('t0', f))
        print("Found ", cols, "series in filelist")
    if rows == None:
        try:
            rows = findSetEnd('t00', f)[1]
        except IndexError:
            rows = findSetEnd('t0', f)[1]

    print(cols,rows)
    fig, axarr = plt.subplots(rows, cols, sharex='all', sharey='all', figsize=(size*cols, size*rows))
    for r, im in enumerate(f):
        if vertPlot:
            j, i = int(r/rows), r%rows
        else:
            i, j = int(r/cols), r%cols
        if rightToLeft:
            j = cols - 1 - j
        a = plt.imread(im)

        if title:
            pos = im.find('Series')
            tit = im[pos:pos+9]
        else: tit = ''

        if rows == 1:
            axarr[j].imshow(a, origin='lower')
            axarr[j].axis('off')
        if cols == 1:
            axarr[i].imshow(a, origin='lower')
            axarr[i].axis('off')

        else:
            axarr[i, j].set_title(tit)
            axarr[i, j].imshow(a, origin='lower')  # Keyword origin='lower' vertically flips the image (microscope image is inverted)
            axarr[i, j].axis('tight')
    # plt.tight_layout(pad=0, h_pad=0)
    if rows == 1:
        fig.set_figwidth(cols*size, size)
        fig.set_figheight(15)

    plt.subplots_adjust(left=lmargin, right=1-lmargin, bottom=lmargin, top=1-lmargin, hspace=gap, wspace=gap/100, )

def find_blobs(f : str,
               ax,
               area : float = 2,
               minTh : float = 200.0,
               figsize : float = 5) :
    """cv2 heuristic algorithm for blob finding
    --------
    Parameters:
    f : str
        image path
    ax : matplotlib.Axes
        axes object to plot on
    area : float
        min blob area
    figsize : float
        figsize, duh
    --------
    Returns:

    """
    # Read image and convert from BGR (cv2) to RGB (python)
    img = cv2.imread(f, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Flip image vertically (acquired inverted from microscope)
    img = cv2.flip( img, 0 )

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Fill any white/yellowish spots with black
    lower_white = np.array([0,200,0])
    upper_white = np.array([255,255,255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    imfil = cv2.bitwise_and(img, img, mask=mask)

    # Invert image for easier contrast in blob finding
    inv = cv2.bitwise_not(imfil)

    ################ Define blob parameters and initialize detector
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = minTh
    params.maxThreshold = 700

    # Change size
    params.filterByArea = True
    params.minArea = area

    params.filterByInertia = False
#     params.minInertiaRatio = 0.001

    params.filterByConvexity = False
    params.filterByCircularity = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inv)
#     print(len(keypoints))
    font = cv2.FONT_HERSHEY_SIMPLEX
    ##### Plot marker on keypoints
    for j, marker in enumerate(keypoints):
        img2 = cv2.drawMarker(imfil, tuple(int(i) for i in marker.pt),
                              (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2 = cv2.putText(img2, str(j), tuple(int(x) for x in marker.pt),
                           fontFace=font, fontScale=0.6,
                           color=(255,255,255), lineType=cv2.LINE_AA)

    if len(keypoints) != 0:
        ax.imshow(img2)
    else:
        ax.imshow(img)
    fig = plt.gcf()
    fig.set_figwidth(figsize)
    fig.set_figheight(figsize)
    sz = [i for i in keypoints]
    return sz

def blob_whole_grid(f: list, cols: int = None, rows: int = None,
                    size: float = 10,
                    area : float = 2,
                   rightToLeft = False, vertPlot = True,
                   lmargin=0.01, gap=0.01,
                   title=False):
    """ Plot whole grid of one channel of images
    f : list -> list of image files to plot
    cols : int -> number of columns in the grid to plot
    rows : int -> number of rows in the grid [set to one for 1D axarr]
    size : float -> image size
    rightToLeft : bool -> flag to plot columns from right to left
    vertPlot : bool -> flag to plot from up to down. If need to plot only one column use this flag
    title : bool -> display image title"""

    if cols == None:
        cols = len(findSetEnd('t00', f))
        print("Found ", cols, " series in filelist")
    if rows == None:
        rows = findSetEnd('t00', f)[1]
        print("Each series contains ", rows, " images")

    fig, axarr = plt.subplots(rows, cols, sharex='all', sharey='all', figsize=(size*cols, size*rows))
    for r, img in enumerate(f):
        if vertPlot:
            j, i = int(r/rows), r%rows
        else:
            i, j = int(r/cols), r%cols
        if rightToLeft:
            j = cols - 1 - j
        if title:
            pos = img.find('Series')
            tit = img[pos:pos+9]
        else: tit = ''

        if (rows == 1) or (cols == 1):
            print(r, img)
            find_blobs(img, axarr[j], area=area, figsize=size)
#             axarr[j].imshow(a, origin='lower')
#             axarr[j].axis('off')

        else:
            axarr[i, j].set_title(tit)
#             axarr[i, j].imshow(a, origin='lower')  # Keyword origin='lower' vertically flips the image (microscope image is inverted)
            find_blobs(img, axarr[i, j], area=area, figsize=size)
            axarr[i, j].axis('tight')
    # plt.tight_layout(pad=0, h_pad=0)
    if (rows == 1) or (cols == 1):
        fig.set_figwidth(cols*size, size)
        fig.set_figheight(15)


    plt.subplots_adjust(left=lmargin, right=1-lmargin,
                        bottom=lmargin, top=1-lmargin,
                        hspace=gap, wspace=gap/100, )

def get_relativeTime(properties_xml : str) -> list:
    """Get relative timestamp of a Leica Series from its Properties.xml file
    Parameters
    ------------
    properties_xml : path
        path and name of SeriesX_Properties.xml file
    """
    with open(properties_xml, 'r') as filebytes:
        lines = filebytes.read()
        tstamp = []

    # Find float pattern attributed to RelativeTime and append to array
        for match in re.finditer('RelativeTime="(\d+\.\d+)"', lines):
    #         print(match.group(1))
            tstamp.append(float(match.group(1)))

    # Include also integers (no decimal point) and then sort by order
        for match in re.finditer('RelativeTime="(\d+)"', lines):
            tstamp.append(float(match.group(1)))

        tstamp = np.array(tstamp)
        tstamp.sort()
        return tstamp


@dataclass
class Bleach:
    """Class to manage and compute bleaching in a time series"""
    path : str
    seriesName : str

    def get_properties_xml(self) -> str:
        return os.path.join(self.path, self.seriesName+'_Properties.xml')

    def get_jpg_names(self) -> list:
        regexp = os.path.join(self.path, self.seriesName + '_*.jpg')
#         print(regexp)
        f = glob.glob(os.path.expandvars(regexp))
        f.sort()
        return f

    def get_relative_time(self) -> list:
        """Get relative timestamp of a Leica Series from its Properties.xml file
        Parameters
        ------------
        properties_xml : path
            path and name of SeriesX_Properties.xml file
        """
        properties_xml = self.get_properties_xml()

        with open(properties_xml, 'r') as filebytes:
            lines = filebytes.read()
            tstamp = []

        # Find float pattern attributed to RelativeTime and append to array
            for match in re.finditer('RelativeTime="(\d+\.\d+)"', lines):
        #         print(match.group(1))
                tstamp.append(float(match.group(1)))

        # Include also integers (no decimal point) and then sort by order
            for match in re.finditer('RelativeTime="(\d+)"', lines):
                tstamp.append(float(match.group(1)))

            tstamp = np.array(tstamp)
            tstamp.sort()
            return tstamp
    def average_drop_bg(self, xcut):
        jpg_files = self.get_jpg_names()

        avDrop = [cv2.mean(cv2.imread(x)[:, xcut:])[0] for x in jpg_files]
        avBg = [cv2.mean(cv2.imread(x)[:, :xcut])[0] for x in jpg_files]

        lenBg = cv2.imread(jpg_files[0])[:, xcut:].size/3
        lenDrop = cv2.imread(jpg_files[0])[:, :xcut].size/3

        davDrop = [cv2.meanStdDev(cv2.imread(x)[:, xcut:])[1][0][0]/lenDrop for x in jpg_files]
        davBg = [cv2.meanStdDev(cv2.imread(x)[:, :xcut])[1][0][0]/lenBg for x in jpg_files]

        return [avDrop, davDrop, avBg, davBg]
