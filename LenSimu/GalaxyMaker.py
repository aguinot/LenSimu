# -*- coding: utf-8 -*-

"""GALAXY MAKER

This file contains methods to simulate a galaxies.

:Author: Axel Guinot

"""

import numpy as np
import galsim
import bisect


class GalaxyMaker(object):
    """Galaxy Maker

    This class contruct a bulge+disk galaxy model from input parameters.
    The main method to call here is "make_gal" which hadle the process.

    Parameters
    ----------
    param_dict: dict
        Dictionary with config values.
        Not used ATM.

    """

    def __init__(self):
        self._n_bins = np.linspace(0.3, 6.2, 21)
        self._n_val = np.mean([self._n_bins[:-1], self._n_bins[1:]], axis=0)

    def get_sersic_index(self, n):
        if n <= 0.3:
            return 0.3

        if n >= 6.2:
            return 6.2

        ind_n = bisect.bisect(self._n_bins, n) - 1
        return self._n_val[ind_n]

    def make_model(self, flux, hlr, q, beta, n):
        """Make disk

        Construct the disk of the galaxy model.
        The disk use an Exponential light profile (sersic n=1).

        Paramters
        ---------
        flux: float
            Flux of the disk.
        hlr: float
            Half-light-radius of the major axis of the disk.
            (hard coded with MICE simu format)
        q: float
            Ratio of the minor axis, b, and the major axis, a, defined as b/a.
        beta: float
            Angle orientation between the major axis and the horizon. Defined
            anti-clock wise. In degrees.

        Returns
        -------
        disk: galsim.GSObject
            Galsim object with the disk model.
        g1, g2: float, float
            Intrinsic ellipticity of the disk. Defined as galsim reduced shear.
        """

        gal_model = galsim.Sersic(
            half_light_radius=hlr,
            flux=flux,
            n=n,
        )
        int_shape = galsim.Shear(q=q, beta=beta * galsim.degrees)
        gal_model = gal_model.shear(int_shape)

        return gal_model, int_shape

    def make_gal(self, flux, hlr, q, beta, n, shear_g1, shear_g2):
        """Make galaxy

        This method handle the creation of the galaxy profile form a sersic.
        The created profile is not convolved by the PSF. The profile have an
        intrinsic ellipticity and a shear.

        Parameters
        ----------
        flux: float
            Total flux of the profile
        hlr: float
            Half-light-radius.
        q: float
            Ratio of the minor axis, b, and the major axis, a, defined as b/a.
        beta: float
            Angle orientation between the major axis and the horizon. Defined
            anti-clock wise. In degrees.
        shear_gamma1, shear_gamma2: float, float
            Shear gamma values.
        kappa: float
            Shear kappa.

        Returns
        -------
        gal: galsim.GSObject
            Galsim object with the galaxy total model.
        intrinsic_g1, intrinsic_g2: float, float
            Intrinsic ellipticity of the full model. Defined as galsim reduced
            shear.

        """

        new_n = self.get_sersic_index(n)
        gal, int_shape = self.make_model(flux, hlr, q, beta, new_n)

        gal = gal.shear(g1=shear_g1, g2=shear_g2)

        intrinsic_g1 = int_shape.g1
        intrinsic_g2 = int_shape.g2

        return (
            gal,
            new_n,
            intrinsic_g1,
            intrinsic_g2,
        )
