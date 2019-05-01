from   dataclasses import dataclass
from  pandas import DataFrame
import pandas as pd
import numpy as np
from  . sbt_types import Molecule
from  . sbt_types import Molecule2P

from   invisible_cities.core.system_of_units import *

GM = 1e-50 * cm2*cm2*second

@dataclass
class FIB(Molecule2P):
    xsTPA : DataFrame = pd.read_csv("/Users/jjgomezcadenas/Projects/Development/sabatsw/data/fluoTPA.csv")
    fibc : DataFrame = pd.read_csv("/Users/jjgomezcadenas/Projects/Development/sabatsw/data/fib_chelated.txt.csv")
    fibu : DataFrame = pd.read_csv("/Users/jjgomezcadenas/Projects/Development/sabatsw/data/fib_unchelated.txt.csv")
    def fibc_spectrum(self, lamda : float)->float:
        return np.interp(lamda, self.fibc.L.values, self.fibc.I.values)
    def fibu_spectrum(self, lamda : float)->float:
        return np.interp(lamda, self.fibu.L.values, self.fibu.I.values)

    def sigma2(self, lamda : float)->float:
        return np.interp(lamda, self.xsTPA.L.values, self.xsTPA.S2.values)

@dataclass
class Fluo3:
    xsTPA : DataFrame = pd.read_csv("/Users/jjgomezcadenas/Projects/Development/sabatsw/data/fluoTPA.csv")

    def sigma2(self, lamda : float)->float:
        return np.interp(lamda, self.xsTPA.L.values, self.xsTPA.S2.values)
