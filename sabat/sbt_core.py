from  . sbt_types import FoV
from  . sbt_types import Laser
from  . sbt_types import PulsedLaser
from  . sbt_types import GLaser
from  . sbt_types import DyeSample
from  . sbt_types import Molecule
from  . sbt_types import Molecule2P
from  . sbt_types import Microscope
from  . sbt_types import CCD

import numpy as np
from  invisible_cities.core.system_of_units import *


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


def duration_of_fluorescence(ml: Molecule, I: float, case :str ='oxygenated')->float:
    """Duration of fluorescence due to photobleaching:
       three cases are considered : oxygenated, deoxigenated, dry

       """
    return ml.max_photons(case) / fluorescence_per_molecule(ml, I)


def diffraction_limit(l : GLaser, mc : Microscope)->float:
    return 1.22 * l.lamda/(2 * mc.numerical_aperture)


def photoelectrons_per_pixel(np : float, ccd : CCD)->float:
    return np / ccd.pixels()


def fluorescence_2p(ds: DyeSample, m: Molecule2P, lb: GLaser, mc: Microscope, n: float = 1)->float:
    """
    Returns the number of photons emitted by fluorescence through 2 photon absorption
    in the focal volume of a strongly focused illumination (e.g, diffraction limited)
    """
    gp = 0.664
    t1 = 0.5 * m.Q * m.sigma2(lb.lamda) * ds.rho_molecules()
    t2 = gp /(lb.f * lb.tau)
    t3 = (n * np.pi * lb.n_photons()**2)/(lb.lamda)
    return   t1 * t2 * t3 * mc.transmission()


def absorbed_photons_per_fluorophore_per_pulse_2p(m: Molecule2P, lb: GLaser, mc : Microscope)->float:
    """
        na = (p0^2 * delta)/(tau * f^2) * (A^2/(2 hbarc * lambda))^2
        natural units, hbarc = 1

    """

    hbarc   = 3.16    * 1e-24                    # J cm
    p0      = lb.power/W                         # W (J/S)
    delta   = m.sigma2(lb.lamda)/(cm2*cm2*s)     # cm4 s
    tau     = lb.tau/second                     # s
    f       = lb.f/hertz                            # hZ
    lamda   = lb.lamda/cm                       # cm
    A       = mc.numerical_aperture

    print(f' p0         = {p0} J/s')
    print(f' Q * delta  = {m.Q * delta} cm4 s/ (molecule photon)')
    print(f' tau * f**2 = {(tau * f**2)} s^-1')
    print(f' hbarc      = {hbarc} J cm')
    print(f' lamda      = {lamda} cm')
    print(f' A          = {A} ')

    t1 = (p0**2 * m.Q * delta) / (tau * f**2)
    t2 = (A**2 / (2 * hbarc * lamda))**2
    print(f' (p0**2 * m.Q * delta) / (tau * f**2) = {t1} J^2 cm^4')
    print(f'((A**2 / (2 * hbarc * lamda))**2       ={t2} J^-2 cm^-4')

    return t1 * t2
