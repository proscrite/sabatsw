from  . sbt_types import FoV
from  . sbt_types import Laser
from  . sbt_types import PulsedLaser
from  . sbt_types import GaussianBeam
from  . sbt_types import CircularFoV
from  . sbt_types import GLaser
from  . sbt_types import DyeSample
from  . sbt_types import Molecule
from  . sbt_types import Molecule2P
from  . sbt_types import Microscope
from  . sbt_types import Monolayer
from  . sbt_types import PhotonsPerSample
from  . sbt_types import CCD
from  .sbt_types import  photon, molecule, GM, us, ucm2, ucm3, gp

import numpy as np
import pandas as pd
import os, sys

from  invisible_cities.core.system_of_units import *
import matplotlib.pyplot as plt

GM3 = 1e-83 * cm**6*second**2


class XMOL:
    def __init__(self, path, tpa_file, fibc_file, fibu_file):

        self.xsTPA = pd.read_csv(os.path.join(path, tpa_file))
        self.fibc  = pd.read_csv(os.path.join(path, fibc_file))
        self.fibu  = pd.read_csv(os.path.join(path, fibu_file))

    def fibc_spectrum(self, lamda : float)->float:
        return np.interp(lamda, self.fibc.L.values, self.fibc.I.values)

    def fibu_spectrum(self, lamda : float)->float:
        return np.interp(lamda, self.fibu.L.values, self.fibu.I.values)

    def sigma2(self, lamda : float)->float:
        return np.interp(lamda, self.xsTPA.L.values, self.xsTPA.S2.values * GM)

    def plot_fibc_spectrum(self):

        fig = plt.figure(figsize=(12,6))
        ax      = fig.add_subplot(1, 1, 1)
        plt.plot(self.fibc.L, self.fibc.I)
        plt.plot(self.fibu.L, self.fibu.I)
        plt.xlabel(r"$\lambda$ (nm)")
        plt.ylabel("I (au)")
        plt.grid(True)

    def plot_TPA(self):

        fig = plt.figure(figsize=(12,6))
        ax      = fig.add_subplot(1, 1, 1)
        plt.plot(self.xsTPA.L, self.xsTPA.S2)
        plt.xlabel(r"$\lambda$ (nm)")
        plt.ylabel(r"$\delta$ (GM)")
        plt.grid(True)

class FIB(XMOL):
    def __init__(self, Q=0.9):
        path       = '/Users/jjgomezcadenas/Projects/Development/sabatsw/data'
        tpa_file   = 'fluoTPA.csv'
        fibc_file  = 'fib_chelated.txt.csv'
        fibu_file  = 'fib_unchelated.txt.csv'
        self.Q     = Q
        XMOL.__init__(self, path, tpa_file, fibc_file, fibu_file)

    def sigma3(self, lamda : float)->float:
        return GM3

    def __repr__(self):
        s ="""
        FIB:
        sigma2  (500 nm)  ={0:5.1f} GM
        sigma3  (500 nm)  ={1:5.1e} GM3
        Q                 ={2:5.1e}

        """.format(self.sigma2(500 * nm) / GM,
                   self.sigma3(500 * nm) / GM3,
                   self.Q
                   )
        return s


class FLUO3(XMOL):
    def __init__(self):
        path       = '/Users/jjgomezcadenas/Projects/Development/sabatsw/data'
        tpa_file   = 'fluoTPA.csv'
        fibc_file  = 'FLUO3_chelated.csv'
        fibu_file  = 'FLUO3_chelated.csv'
        self.Q     = 0.9

        XMOL.__init__(self, path, tpa_file, fibc_file, fibu_file)


class Setup:
    def __init__(self,  setup_name             = 'Espinardo',
                        molecule_name          = 'FLUO3',
                        sample_name            = '5ba',
                        sample_concentration   = 1E+6/6 * nanomole/liter,
                        laser_lambda           = 800 * nm,
                        laser_power            = 100 * mW,
                        laser_eff              = 1,
                        laser_lambda_eff       = 1,
                        laser_f                = 76  * megahertz,
                        laser_tau              = 400 * femtosecond,
                        mc_name                = 'JMB',
                        mc_NA                  = 0.5,
                        mc_M                   = 20,
                        mc_eff_dic             = 0.7,
                        mc_eff_filt            = 1.0,
                        mc_eff_PMT             = 0.1,
                        n_pixels               = 256,
                        scan_length            = 90 * mum,
                        v_per_line             = 40,
                        f_mirror               = 0.3,
                        out                    = 'verbose'):

        if molecule_name == 'Fluorescein' or molecule_name == 'FLUO3':

            self.fl2 = FLUO3()
        else:
            self.fl2 = FIB()

        if out == 'vverbose':
            self.fl2.plot_fibc_spectrum()
            self.fl2.plot_TPA()

        self.lb = PulsedLaser(lamda = laser_lambda,
                         power = laser_power * laser_eff * laser_lambda_eff,
                         f     = laser_f,
                         tau   = laser_tau)
        if out == 'verbose':
            print(self.lb)
        self.mc = Microscope(name=mc_name, numerical_aperture=mc_NA,
                             magnification=mc_M, eff_dichroic = mc_eff_dic,
                             eff_filter = mc_eff_filt, eff_PMT = mc_eff_PMT)
        if out == 'verbose':
            print(self.mc)

        self.gb = GaussianBeam(laser=self.lb, mc = self.mc)
        if out == 'verbose':
            print(self.gb)

        dl = diffraction_limit(self.lb, self.mc)
        if out == 'verbose':
            print(f' Diffraction limit transverse size of beam = {dl/mum}')


        self.pixel_size_mu_a = scan_length / n_pixels
        if out == 'verbose':
            print(f'pixel size = {self.pixel_size_mu_a/mum:5.1e} mum')

        self.t_line = (1/v_per_line) * second
        self.t_pixel = self.t_line/n_pixels * f_mirror
        if out == 'verbose':
            print(f'time per pixel ={self.t_pixel/mus} (mus)')

        fov= CircularFoV(d = 2* self.gb.w0(), z= 2 * self.gb.zr())
        if out == 'verbose':
            print(fov)

        self.ds = DyeSample(name=sample_name,concentration = sample_concentration,
                            volume= fov.volume())
        if out == 'verbose':
            print(self.ds)

        if out == 'verbose':
            print(f'Fluorescence (n= 2) ={self.fluorescence(n_photons=2)/us:5.1e}')
            print(f'Fluorescence (n= 3) ={self.fluorescence(n_photons=3)/us:5.1e}')


    def fluorescence(self, n_photons=2)->float:
        if n_photons == 2:
            F = fluorescence_2p_dl(self.ds, self.fl2, self.lb, self.mc)
        elif n_photons == 3:
            F = fluorescence_3p_dl(self.ds, self.fl2, self.lb, self.mc)
        else:
            print(f'not implemented')
            F = 0
        return F


    def photons_per_pixel(self, n_photons)->float:
        F2 = self.fluorescence(n_photons)/us
        F_pixel = F2 * (self.t_pixel/mus) * 1e-6
        return F_pixel

    def photons_per_pixel_per_molecule(self, n_photons)->float:
        F2 = self.fluorescence(n_photons)/us
        F_pixel = F2 * (self.t_pixel/mus) * 1e-6
        return self.photons_per_pixel(n_photons) / self.ds.n_molecules()

    def detected_photons_per_pixel(self, n_photons)->float:
        n_f = self.photons_per_pixel(n_photons) * self.mc.transmission()
        return n_f


def signal(ml : Monolayer, n_exp : int =int(1e+2))->np.array:
    mu_s    = ml.nf
    sigma_s = np.sqrt(mu_s)
    alpha   = ml.alpha
    m       = int(ml.n_molecules)
    mu_b = mu_s  / alpha
    sigma_b = np.sqrt(mu_b)

    N = []
    for i in range(n_exp):
        n_s = np.random.normal(mu_s, sigma_s)
        n_b = np.sum(np.random.normal(mu_b, sigma_b, m))
        nt = n_s + n_b
        n = nt - mu_b * m
        N.append(n /mu_s)
    return N


def photon_per_sample(n_f :float, readout_f : float, alpha : float, mc: Microscope)->PhotonsPerSample :
    ns_ph   = n_f * readout_f
    ns_det  = ns_ph * mc.transmission()
    nb_ph   = ns_ph / alpha
    nb_det  = ns_det/ alpha
    return PhotonsPerSample(ns_ph, ns_det, nb_ph, nb_det)


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


def fluorescence_2p_gb(ds: DyeSample,
                       m : Molecule2P,
                       lb: GLaser,
                       mc: Microscope,
                       n: float = 1)->float:
    """
    Returns the number of photons emitted by fluorescence through 2 photon absorption
    in the focal volume of a gaussian beam (e.g, beam narrower than lens)
    """
    gp = 0.664

    t1 = 0.5 * m.Q * m.sigma2(lb.lamda) * ds.rho_molecules()
    t2 = gp /(lb.f * lb.tau)
    t3 = (n * np.pi * lb.n_photons()**2)/(lb.lamda)
    return   t1 * t2 * t3 * mc.transmission()


def fluorescence_2p_dl(ds      : DyeSample,
                       m       : Molecule2P,
                       lb      : GLaser,
                       mc      : Microscope,
                       n       : float = 1,
                       verbose : bool  = False)->float:
    """
    Returns the number of photons emitted by fluorescence through 2 photon absorption
    in the focal volume of a strongly focused illumination (e.g, diffraction limited)
    """
    gp = 0.664
    a0 = 64
    t1 =  m.sigma2(lb.lamda) * ds.rho_molecules() * lb.n_photons()**2/lb.lamda
    t2 = gp /(lb.f * lb.tau)
    t3 = m.Q * n * a0 /(16 * np.pi)

    if verbose:

        print(f' Q = {m.Q}, sigma2 ({lb.lamda/nm:7.2f}) = {m.sigma2(lb.lamda)/GM:7.2f} GM')
        print(f' rho = {ds.rho_molecules()/ucm3:5.1e} molecules/cm3')
        print(f' f * tau = {lb.f * lb.tau:5.1e}')
        print(f' photons/second = {lb.n_photons():5.1e}')
        print(f' t1 =  C * sigma * P^2 / lamda = {t1/us:5.1e} s' )
        print(f' t2 = gp/(f tau) = {t2:5.1e}' )
        print(f' t3 = A * Q = {t3:5.1e}' )
        print(f' microscope T = {mc.transmission():5.1e}' )
    return   t1 * t2 * t3 * mc.transmission()


def fluorescence_3p_dl(ds: DyeSample,
                       m : Molecule2P,
                       lb: GLaser,
                       mc: Microscope,
                       n: float = 1)->float:
    """
    Returns the number of photons emitted by fluorescence through 3 photon absorption
    in the focal volume of a strongly focused illumination (e.g, diffraction limited)
    """
    gp = 0.41
    t1 = (1/3) * m.Q * m.sigma3(lb.lamda) * ds.rho_molecules()
    t2 = gp /(lb.f * lb.tau)**2
    t3 = (n * 3.5 * lb.n_photons()**3 * mc.numerical_aperture**2)/(lb.lamda **3)
    return   t1 * t2 * t3 * mc.transmission()



def fluorescence_dl(ds         : DyeSample,
                    m          : Molecule2P,
                    lb         : GLaser,
                    mc         : Microscope,
                    n_photons  : float =2,
                    n0         : float = 1,
                    verbose    : bool  = False)->float:
    """
    Returns the number of photons emitted by fluorescence through n photon absorption
    in the focal volume of a strongly focused illumination (e.g, diffraction limited)
    """
    def print_():
        print(f' number of photons absorbed ={n}')
        print(f' Q = {m.Q}, sigma ({lb.lamda/nm:7.2f}) = {sigma/G:7.2f} GM')
        print(f' rho = {ds.rho_molecules()/ucm3:5.1e} molecules/cm3')
        print(f' f * tau = {lb.f * lb.tau:5.1e}')
        print(f' photons/second = {lb.n_photons():5.1e}')
        print(f' t1 =  C * sigma * P^2 / lamda = {t1/us:5.1e} s' )
        print(f' t2 = gp/(f tau) = {t2:5.1e}' )
        print(f' t3 = A * Q = {t3:5.1e}' )
        print(f' microscope T = {mc.transmission():5.1e}' )

    if n_photons == 2:
        gp = 0.664
        a0 = 64
        sigma = m.sigma2(lb.lamda)
        G = GM
    elif n_photons == 3:
        gp = 0.51
        a0 = 28.1
        sigma = m.sigma3(lb.lamda)
        G = GM3
    else:
        print('not implemented')
        sys.exit()

    n = n_photons

    t1 =  sigma * ds.rho_molecules() * lb.n_photons()**n/(lb.lamda**(2*n-3))
    t2 = gp /(lb.f * lb.tau)**(n-1)
    t3 = (1/n) * (a0 * mc.numerical_aperture**(2*n - 4)/(8*np.pi**(3 - n)))* m.Q

    if verbose:
        print_()

    return   t1 * t2 * t3 * mc.transmission()


def fluorescence_dl_2p(ds         : DyeSample,
                       m          : Molecule2P,
                       lb         : GLaser,
                       mc         : Microscope,
                       n0         : float = 1,
                       verbose    : bool  = False)->float:
    """
    Returns the number of photons emitted by fluorescence through 2 photon absorption
    in the focal volume of a strongly focused illumination (e.g, diffraction limited).
    Takes into account saturation (e.g, max number of absorbed photons per molecule)
    """

    na = absorbed_photons_per_fluorophore_per_pulse_2p(m, lb, mc)
    #print(f' na ={na} ')
    if na >= 2:
        #print(lb.f* mc.transmission()/us)
        return lb.f* mc.transmission()
    else:
        #print(fluorescence_dl(ds, m, lb, mc, n_photons = 2, verbose=verbose)/us)
        return fluorescence_dl(ds, m, lb, mc, n_photons = 2, verbose=verbose)



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

    # print(f' p0         = {p0} J/s')
    # print(f' Q * delta  = {m.Q * delta} cm4 s/ (molecule photon)')
    # print(f' tau * f**2 = {(tau * f**2)} s^-1')
    # print(f' hbarc      = {hbarc} J cm')
    # print(f' lamda      = {lamda} cm')
    # print(f' A          = {A} ')

    t1 = (p0**2 * m.Q * delta) / (tau * f**2)
    t2 = (A**2 / (2 * hbarc * lamda))**2
    # print(f' (p0**2 * m.Q * delta) / (tau * f**2) = {t1} J^2 cm^4')
    # print(f'((A**2 / (2 * hbarc * lamda))**2       ={t2} J^-2 cm^-4')

    return t1 * t2
