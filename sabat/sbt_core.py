from  . sbt_types import FoV
from  . sbt_types import LaserBeam
from  . sbt_types import PulsedBeam
from  . sbt_types import Molecule
from  . sbt_types import Molecule2P
from  . sbt_types import Microscope
from  . sbt_types import CCD
from  . invisible_cities.core.system_of_units import *


def power_density(lb : GLaser, fov : FoV)->float:
    return lb.power / fov.area()


def photon_density(lb : GLaser, fov : FoV)->float:
    return power_density(lb, fov) / lb.photon_energy()


def fluorescence_per_molecule(m: Molecule, I: float)->float:
    """
    Returns the number of photons per molecule emitted by fluorescence.
    Parameters:
        m     : defines molecule
        I     : photon density (nphot/cm2)

    """
    return m.sigma * m.Q * I


def duration_of_fluorescence(ml: Molecule, I: float, time: float, case :str ='oxygenated')->float:
    """Duration of fluorescence due to photobleaching:
       three cases are considered : oxygenated, deoxigenated, dry

       """
    return ml.max_photons(case) / fluorescence_per_molecule(ml, I, time) * second


def diffraction_limit(l : LaserBeam, mc : Microscope)->float:
    return l.lamda/(2 * mc.numerical_aperture)


def photoelectrons_per_pixel(np : float, ccd : CCD)->float:
    return np / ccd.pixels()
