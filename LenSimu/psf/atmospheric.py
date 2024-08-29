"""
Atmospheric PSF

Here we create an atmospheric PhaseScreen to describe the variation of the PSF
model due to atmospheric turbulence. Then, we can draw PSF models from the
instantiated class.

"""

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.stats import rv_histogram

import galsim

from ..utils import parser
from .optical_psf import optical


# Big number to represent "infinite" values
_BIG_NUMBER = 1e30
# N sample to integrate Cn2
_N_SAMPLE = 1000
# Max altitude to consider to compute screen size (in m)
_MAX_ALT = 30e3
#
_SCREEN_SCALE = 0.1


class atmosphere:
    """

    Parameters
    ----------
    config: str, dict
        Path to the Yaml config file or an instantiate dict.
    lam: float
        Wavelength (in nm).
    theta: float
            Zenith angle (in deg).
            Angle between vertical and pointing.
            [Default: 0.]
    """

    def __init__(
        self,
        config,
        lam,
        theta=0.0,
        target_seeing=None,
        seed=None,
        full_atm=True,
    ):
        """ """

        self._lam = lam

        # Read config
        self._atm_config, self._opt_config = self._load_config(config)

        # Init randoms
        self._rng = np.random.RandomState(seed=seed)
        self._galsim_rng = galsim.BaseDeviate(seed=seed + 4)

        # Init L0
        self._get_L0()

        # Init Cn2 coeffs
        self._Cn2_coeff = self._atm_config["HV_coeff"]

        # Get Cn2_dh
        self._get_Cn2_dh()

        # Get r0_500
        self.r0_500 = self._get_r0(500e-9, theta=theta)

        # Get wind speed and direction
        self.wind_speed, self.wind_dir = self._get_wind(
            self.alts, self._n_layers
        )

        # Get screen size
        self._get_screen_size()

        self._target_seeing = target_seeing
        self._full_atm = full_atm

    def _load_config(self, config):
        """Load config

        Load the config file for the PSF and keep only the atmospheric part.
        Check if all the needed values are presents.

        Parameters
        ----------
        config_path: str
            Path to the Yaml config file.

        Return
        ------
        config: dict
            Dictionary parsed from the config file.

        """

        if isinstance(config, str):
            config_dict = parser.ReadConfig(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise ValueError(
                "config must be a path to a config Yaml file or an instantiate"
                " dictionary."
            )

        parser._check_config(config_dict, parser._config_atmo_template)

        return (config_dict["atmospheric"], config_dict["telescope"])

    def _get_L0(self):
        """Get L0

        Load L0 value and altitudes.

        """

        self.alts = np.array(self._atm_config["L0_values"]["alts"])
        L0 = self._atm_config["L0_values"]["L0"]
        if len(self.alts) != len(L0):
            raise ValueError(
                'The size of "alts" and "L0" should match got: '
                + f"{len(self.alts)} and {len(L0)} respectively"
            )
        self._n_layers = len(self.alts)

        if self._atm_config["L0_values"]["spread"]:
            if "sig" not in self._atm_config["L0_values"].keys():
                raise ValueError(
                    '"sig" key not found in "L0_values" section'
                    + "in config ."
                )
            sig = self._atm_config["L0_values"]["sig"]
            self.L0 = self._rng.lognormal(
                mean=np.log(L0),
                sigma=sig,
                size=(1, self._n_layers),
            ).squeeze()
        else:
            self.L0 = np.array(L0)

    def _get_Cn2(self, h):
        """Get Cn2

        Get Cn2 values from Hufnagel-Valley model.
        The values are returned in m^(-2/3)
        https://arxiv.org/abs/astro-ph/0202137
        https://books.google.fr/books?hl=fr&lr=&id=-0aAWyckS_8C&oi=fnd&pg=PA3&dq=Hardy,+J.+W.,+ed.+1998,+Adaptive+optics+for+astronomical+telescopes+(Oxford+University+Press)&ots=k98e7Ip_Nn&sig=2hJhKTN3qBcIXs8Q6zNkzlLRze8#v=onepage&q&f=false  # noqa

        Parameters
        ----------
        h: float
            Altitude at which we want Cn2.

        Return
        ------
        Cn2: float
            Cn2 at altitude h.

        """
        # Surface Layer
        Cn_surf = self._Cn2_coeff["A0"] * np.exp(-h / self._Cn2_coeff["H0"])
        # Ground Layer
        Cn_gr = self._Cn2_coeff["A1"] * np.exp(-h / self._Cn2_coeff["H1"])
        # Tropopause Layer
        Cn_trop = (
            self._Cn2_coeff["A2"] * h**10 * np.exp(-h / self._Cn2_coeff["H2"])
        )
        # Strong layer (Mid altitudes)
        Cn_str = self._Cn2_coeff["A3"] * np.exp(
            -((h - self._Cn2_coeff["H3"]) ** 2.0)
            / (2.0 * self._Cn2_coeff["d"] ** 2.0)
        )

        # Total
        return Cn_surf + Cn_gr + Cn_trop + Cn_str

    def _get_Cn2_dh(self):
        """Get Cn2_dh

        Integrate the Cn2 values for each layers.

        NOTE:
        Sampling the altitudes in log space showed better results.

        """

        n_alts = len(self.alts)
        Cn2_dh = []
        for i in range(n_alts):
            start = self.alts[i]
            if self.alts[i] == 0.0:
                start = 0
            else:
                start = np.log10(self.alts[i])
            if i < n_alts - 1:
                end = np.log10(self.alts[i + 1])
            else:
                # Infinite altitude
                end = np.log10(_BIG_NUMBER)

            alts = np.logspace(start, end, _N_SAMPLE)
            Cn2_alts = self._get_Cn2(alts)
            Cn2_dh.append(simpson(Cn2_alts, x=alts))

        self.Cn2_dh = np.array(Cn2_dh)

    def _get_r0(self, lam, theta=0.0):
        """Get r0

        Compute Fried Parameter.
        https://arxiv.org/abs/astro-ph/0202137

        Parameters
        ----------
        lam: float
            Wavelength (in m).
        theta: float
            Zenith angle (in deg).
            Angle between vertical and pointing.
            [Default: 0.]

        Return
        ------
        r0: float
            Fried parameter.

        """

        k = 2.0 * np.pi / lam
        r0_vert = (0.423 * k**2.0 * self.Cn2_dh) ** (-3 / 5)

        return np.cos(theta * np.pi / 180.0) ** (3 / 5.0) * r0_vert

    def _get_target_seeing(self, target_seeing, do_optical=True):
        if do_optical:
            opt_psf = self.opt.get_optical_psf(
                pupil_bin=8,
                pad_factor=2,
            )
        else:
            opt_psf = galsim.Airy(
                lam=self._lam,
                diam=self._opt_config["aperture"]["diam"],
                obscuration=self._opt_config["aperture"]["obscuration"],
            )

        star = galsim.DeltaFunction().withFlux(1e5)

        def chi2(r0_factor):
            L0, r0_500, _ = self._get_L0_r0_eff(r0_factor=r0_factor)
            model_ = self.make_VonKarman(L0=L0, r0_500=r0_500)
            model = galsim.Convolve((opt_psf, model_))
            model = galsim.Convolve((model, star))
            model_seeing = (
                model.drawImage(scale=0.187)
                .FindAdaptiveMom(use_sky_coords=True)
                .moments_sigma
                * 2.355
            )
            return (target_seeing - model_seeing) ** 2

        res = minimize(chi2, x0=0.2, bounds=[(1e-2, 5)], method="Nelder-Mead")

        if res.success:
            return res.x
        else:
            print("Failed to get target PSF. r0_factor=1")
            return 1.0

    def _get_L0_r0_eff(self, r0_factor=1.0):
        """Get L0 and r0 effective

        Compute the effective outer scale L0 and Fried parameter at 500nm
        r0_500.

        Parameters
        ----------
        r0_factor : float, optional
            Multiplicative factor to apply to r0 to match a target seeing, by
            default 1.

        Returns
        -------
        tuple
            L0 and r0 effective.
        """

        L0_eff = (
            simpson(self.L0 ** (5 / 3) * self.Cn2_dh, x=self.alts)
            / simpson(self.Cn2_dh, x=self.alts)
        ) ** (3 / 5)

        r0_500_eff = (
            sum(r ** (-5.0 / 3) for r in self._get_r0(500 * 1e-9) * r0_factor)
        ) ** (-3.0 / 5)

        alt_eff = (
            simpson(self.alts ** (5 / 3) * self.Cn2_dh, x=self.alts)
            / simpson(self.Cn2_dh, x=self.alts)
        ) ** (3 / 5)

        return L0_eff, r0_500_eff, alt_eff

    def _get_wind(self, alts, n_layers, theta=0.0):
        """Get wind speed

        Wind profile from Greenwood model.
        Wind direction is set randomly for each layers in [0., 360.].
        Ref: Greenwood (1977)

        Parameters
        ----------
        theta: float
            Zenith angle (in deg).
            Angle between vertical and pointing.
            [Default: 0.]

        """

        coeff = self._atm_config["wind"]["GW_coeff"]
        ground_speed = self._atm_config["wind"]["wind_speed"]["ground"]
        trop_speed = self._atm_config["wind"]["wind_speed"]["trop"]

        v_g = self._rng.uniform(*ground_speed, size=n_layers)
        v_t = self._rng.uniform(*trop_speed, size=n_layers)

        wind_speed = v_g + v_t * np.exp(
            -(
                (
                    (alts * np.cos(theta * np.pi / 180.0) - coeff["H"])
                    / coeff["T"]
                )
                ** 2.0
            )
        )

        wind_dir = self._rng.uniform(0.0, 360.0, size=n_layers)
        wind_dir = wind_dir * galsim.degrees

        return wind_speed, wind_dir

    def _get_screen_size(self, max_alt=None):
        """Get screen size

        Compute the size of simulated screen based on telescope FOV.

        """

        FOV = self._opt_config["FOV"]
        if max_alt is None:
            max_alt = _MAX_ALT

        self._screen_size = 2.0 * max_alt * np.tan(FOV / 2.0 * np.pi / 180.0)

    def _get_sig_mean_atm(self, mean_seeing):
        r0_factor_mean = self._get_target_seeing(mean_seeing)

        L0_mean, r0_500_mean, _ = self._get_L0_r0_eff(r0_factor_mean)

        atm_psf = self.make_VonKarman(L0=L0_mean, r0_500=r0_500_mean)

        return atm_psf.calculateMomentRadius()

    def make_atmosphere(self, gsparams=None, **kwargs):
        """Make atmosphere

        Create galsim object `galsim.Atmosphere` from which we draw PSFs.

        """

        self.atm = galsim.Atmosphere(
            r0_500=self.r0_500 * self._r0_factor,
            altitude=self.alts / 1_000.0,
            L0=self.L0,
            speed=self.wind_speed,
            direction=self.wind_dir,
            screen_size=self._screen_size,
            screen_scale=_SCREEN_SCALE,
            rng=self._galsim_rng,
            gsparams=gsparams,
            **kwargs,
        )

        return self.atm

    def make_simple_atmosphere2(self, **kwargs):
        """Make simple atmosphere

        Create galsim object `galsim.Atmosphere` from which we draw PSFs.

        Same as `make_atmosphere` but using one layer with the effective value
        to speed up the computation. This approach is still slower than the
        Von Karman model so we are not using it at the moment.

        """

        self._get_screen_size(max_alt=self.alt_eff)
        wind_speed_eff, wind_dir = self._get_wind(self.alt_eff, 1)

        self.atm = galsim.Atmosphere(
            r0_500=np.atleast_1d(self.r0_500_eff) * self._r0_factor,
            altitude=np.atleast_1d(self.alt_eff) / 1_000,
            L0=np.atleast_1d(self.L0_eff),
            speed=wind_speed_eff,
            direction=wind_dir,
            screen_size=self._screen_size,
            screen_scale=_SCREEN_SCALE,
            rng=self._galsim_rng,
            **kwargs,
        )

        return self.atm

    def make_simple_atmosphere(self):
        """

        Implementation taken from descwl-shear-sims:
        https://github.com/LSSTDESC/descwl-shear-sims/blob/master/descwl_shear_sims/psfs/ps_psf.py
        """

        trunc = 1.0
        variation_factor = 1

        # We re-initialize
        self._get_screen_size(max_alt=self.alt_eff)

        ng = self._atm_config["simple_model"]["grid_size"]
        # Screen scale in m (we work in m because the Pk is defined in m due to
        # L0_eff being in m)
        gs = self._screen_size * 1.1 / ng

        def _pk(k):
            return (k**2 + (1.0 / self.L0_eff) ** 2) ** (-11.0 / 6.0) * np.exp(
                -((k * trunc) ** 2)
            )

        ps = galsim.PowerSpectrum(
            e_power_function=_pk,
            b_power_function=_pk,
        )

        ps.buildGrid(
            grid_spacing=gs,
            ngrid=ng,
            get_convergence=True,
            variance=(0.01 * variation_factor) ** 2,
            rng=self._galsim_rng,
        )

        g1_grid, g2_grid, mu_grid = galsim.lensing_ps.theoryToObserved(
            ps.im_g1.array, ps.im_g2.array, ps.im_kappa.array
        )

        self._lut_g1 = galsim.table.LookupTable2D(
            ps.x_grid,
            ps.y_grid,
            g1_grid.T,
            edge_mode="wrap",
            interpolant=galsim.Lanczos(5),
        )
        self._lut_g2 = galsim.table.LookupTable2D(
            ps.x_grid,
            ps.y_grid,
            g2_grid.T,
            edge_mode="wrap",
            interpolant=galsim.Lanczos(5),
        )
        self._lut_mu = galsim.table.LookupTable2D(
            ps.x_grid,
            ps.y_grid,
            mu_grid.T - 1,
            edge_mode="wrap",
            interpolant=galsim.Lanczos(5),
        )

        # 0.025 correspond to the variance of PSF ellipticity due to
        # atmosphere variation.
        self._g1_mean = self._rng.normal() * 0.025 * variation_factor
        self._g2_mean = self._rng.normal() * 0.025 * variation_factor

    def _get_lensing(self, theta):
        u = 0.0
        if theta[0].rad != 0.0:
            u = self.alt_eff * theta[0].tan()
        v = 0.0
        if theta[1].rad != 0.0:
            v = self.alt_eff * theta[1].tan()

        return (
            self._lut_g1(u, v),
            self._lut_g2(u, v),
            self._lut_mu(u, v) + 1,
        )

    def make_VonKarman(self, L0=None, r0_500=None, gsparams=None):
        """Make VonKarman

        Create a VonKarman PSF for the current atmosphere. This can be useful
        For extremely bright objects for which drawing with photon shooting
        or using fourier optics on PhaseScreen PSF would be to long.
        It uses an effective r0_500: np.sum(r0_500s**(-5./3))**(-3./5)

        ..math::
            r_{0,500}^{eff} = sum

        and a weighted averaged for L0:

        Parameters
        ----------
        L0: float
            Effective Outer scale in meters, L0
        r0_500: float
            Effective Fried parameter at 500nm, r0_500

        Return
        ------
        psf_VK: galsim.vonkarman.VonKarman
            Galsim VonKarman PSF.

        """

        if L0 is None:
            L0 = self.L0_eff
        if r0_500 is None:
            r0_500 = self.r0_500_eff

        psf_VK = galsim.VonKarman(
            lam=self._lam,
            r0_500=r0_500,
            L0=L0,
            gsparams=gsparams,
        )

        return psf_VK

    def make_Kolmogorov(self, r0_500=None, gsparams=None):
        """Make Kolmogorov

        Create a Kolmogorov PSF for the current atmosphere. This can be useful
        For extremely bright objects for which drawing with photon shooting
        or using fourier optics on PhaseScreen PSF would be to long.
        It uses an effective r0_500: np.sum(r0_500s**(-5./3))**(-3./5)

        ..math::
            r_{0,500}^{eff} = sum

        Parameters
        ----------
        r0_500: float
            Effective Fried parameter at 500nm, r0_500

        Return
        ------
        psf_VK: galsim.vonkarman.VonKarman
            Galsim VonKarman PSF.

        """

        if r0_500 is None:
            r0_500 = self.r0_500_eff

        psf_Kolmo = galsim.Kolmogorov(
            lam=self._lam,
            r0_500=r0_500,
            gsparams=gsparams,
        )

        return psf_Kolmo

    def init_PSF(
        self,
        do_optical=True,
        gsparams=None,
    ):
        if do_optical:
            self.opt = optical(self._opt_config, self._lam)
        else:
            self.aper = None

        # Get target seeing
        self._r0_factor = 1.0
        if isinstance(self._target_seeing, float) | isinstance(
            self._target_seeing, int
        ):
            self._r0_factor = self._get_target_seeing(
                self._target_seeing, do_optical=do_optical
            )

        # Get effective values
        self.L0_eff, self.r0_500_eff, self.alt_eff = self._get_L0_r0_eff(
            r0_factor=self._r0_factor
        )

        if self._full_atm:
            self.SL = self.make_atmosphere(gsparams=gsparams)
        else:
            self.make_simple_atmosphere()
            self._atm_psf = self.make_VonKarman(gsparams=gsparams)
            # self._atm_psf = self.make_Kolmogorov(gsparams=gsparams)
            # self.SL = self.make_simple_atmosphere2()

    def makePSF(
        self,
        theta,
        exptime,
        time_step=10,
        do_optical=True,
        do_atmosphere=True,
        img_pos=None,
        ccd_num=None,
        do_rot=False,
        pupil_bin=16,
        pad_factor=4,
        gsparams=None,
        **atm_kwargs,
    ):
        if not (do_optical or do_atmosphere):
            raise ValueError(
                "Either do_optical or do_atmosphere must be true."
            )

        aper = None
        if do_optical:
            # Here we build pure diffraction PSF (no aberrations)
            opt_psf = self.opt.get_optical_psf(
                pupil_bin=pupil_bin,
                pad_factor=pad_factor,
                gsparams=gsparams,
            )
            fp_g1, fp_g2, fp_size_factor = self.opt.get_focal_plane_value(
                img_pos[0],
                img_pos[1],
                ccd_num,
                do_rot,
            )
            aper = self.opt.aper
            if not do_atmosphere:
                total_psf = opt_psf

        if do_atmosphere:
            # Here we build pure atmospheric PSF
            if self._full_atm:
                atm_psf = self.SL.makePSF(
                    lam=self._lam,
                    theta=theta,
                    aper=aper,
                    exptime=exptime,
                    time_step=time_step,
                    gsparams=gsparams,
                    **atm_kwargs,
                )

            else:
                atm_psf = self._atm_psf
                g1, g2, mu = self._get_lensing(theta)
                if g1 * g1 + g2 * g2 >= 1.0:
                    norm = np.sqrt(g1 * g1 + g2 * g2) / 0.5
                    g1 /= norm
                    g2 /= norm
                atm_psf = atm_psf.shear(
                    g1=g1 + self._g1_mean, g2=g2 + self._g2_mean
                )
                atm_psf = atm_psf.dilate(1 / np.power(mu, 0.75))

            if do_optical:
                total_psf = galsim.Convolve((atm_psf, opt_psf))
            else:
                total_psf = atm_psf

        if do_optical:
            # Finally we apply distortions due to optical aberrations
            # Those aberrations contain atmospheric effect which why we apply
            # them at the end, on the convolved psf_opt, psf_atm.
            total_psf = total_psf.shear(galsim.Shear(g1=fp_g1, g2=fp_g2))
            total_psf = total_psf.dilate(fp_size_factor)
        return total_psf


class seeing_distribution(object):
    """Seeing distribution

    Provide a seeing following CFIS distribution. Seeing generated from
    scipy.stats.rv_histogram(np.histogram(obs_seeing)). Object already
    initialized and saved into a numpy file.

    Parameters
    ----------
    path_to_file: str
        Path to the numpy file containing the scipy object.
    seed: int
        Seed for the random generation. If None rely on default one.

    """

    def __init__(self, path_to_file, seed=None):
        self._file_path = path_to_file
        self._load_distribution()

        self._random_seed = None
        if seed is not None:
            self._random_seed = np.random.RandomState(seed)

    def _load_distribution(self):
        """Load distribution

        Load the distribution from numpy file.

        """
        all_fwhm = np.load(self._file_path)
        m_good_fwhm = (all_fwhm > 0.4) & (all_fwhm < 1.0)
        self._distrib = rv_histogram(
            np.histogram(all_fwhm[m_good_fwhm], 100, density=True)
        )
        self.mean_seeing = np.mean(all_fwhm[m_good_fwhm])

    def get(self, size=None):
        """Get

        Return a seeing value from the distribution.

        Parameters
        ----------
        size: int
            Number of seeing value required.

        Returns
        -------
        seeing: float (numpy.ndarray)
            Return the seeing value or a numpy.ndarray if size != None.

        """

        return self._distrib.rvs(size=size, random_state=self._random_seed)
