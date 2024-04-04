"""
Atmospheric PSF

Here we create an atmospheric PhaseScreen to describe the variation of the PSF
model due to atmospheric turbulences. Then, we can draw PSF models from the
instanciated class.

"""

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize

import galsim

from ..utils import parser

# Big number to represent "infite" values
_BIG_NUMBER = 1e30
# N sample to integrate Cn2
_N_SAMPLE = 1000
# Max altidude to consider to compute screen size (in m)
_MAX_ALT = 30e3
#
_SCREEN_SCALE = 0.1


class atmosphere():
    """

    Parameters
    ----------
    config: str, dict
        Path to the Yaml config file or an instanciate dict.
    lam: float
        Wavelenght (in nm).
    theta: float
            Zenith anlge (in deg).
            Angle between vertical and pointing.
            [Default: 0.]
    """

    def __init__(self, config, lam, theta=0., target_seeing=None, seed=None):
        """
        """

        self._lam = lam

        # Read config
        self._config, self._FOV = self._load_config(config)

        # Init randoms
        self._rng = np.random.RandomState(seed=seed)
        self._galsim_rng = galsim.BaseDeviate(seed=seed)

        # Init L0
        self._get_L0()

        # Init Cn2 coeffs
        self._Cn2_coeff = self._config['HV_coeff']

        # Get Cn2_dh
        self._get_Cn2_dh()

        # Get r0_500
        self.r0_500 = self._get_r0(500e-9, theta=theta)

        # Get wind speed and direction
        self._get_wind()

        # Get screen size
        self._get_screen_size()

        # Get target seeing
        self.r0_factor = 1.
        if (target_seeing is float) | (target_seeing is int):
            self._r0_factor = self._get_target_seeing(target_seeing)

        # Get effective values
        self.L0_eff, self.r0_500_eff = self._get_L0_r0_eff(
            r0_factor=self._r0_factor
        )

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
            Dictionnary parsed from the config file.

        """

        if isinstance(config, str):
            config_dict = parser.ReadConfig(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise ValueError(
                "config must be a path to a config Yaml file or an instanciate"
                + "dictionary."
                )

        parser._check_config(config_dict, parser._config_atmo_template)

        return (
            config_dict['atmospheric'],
            config_dict['telescope']['aperture']['FOV'])

    def _get_L0(self):
        """Get L0

        Load L0 value and altitudes.

        """

        self.alts = np.array(self._config['L0_values']['alts'])
        L0 = self._config['L0_values']['L0']
        if len(self.alts) != len(L0):
            raise ValueError(
                'The size of "alts" and "L0" should match got: '
                + f'{len(self.alts)} and {len(L0)} respectively'
                )
        self._n_layers = len(self.alts)

        if self._config['L0_values']['spread']:
            if 'sig' not in self._config['L0_values'].keys():
                raise ValueError(
                    '"sig" key not found in "L0_values" section'
                    + 'in config .'
                    )
            sig = self._config['L0_values']['sig']
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
        Cn_surf = self._Cn2_coeff['A0']*np.exp(-h/self._Cn2_coeff['H0'])
        # Ground Layer
        Cn_gr = self._Cn2_coeff['A1']*np.exp(-h/self._Cn2_coeff['H1'])
        # Tropopause Layer
        Cn_trop = self._Cn2_coeff['A2']*h**10*np.exp(-h/self._Cn2_coeff['H2'])
        # Strong layer (Mid altitudes)
        Cn_str = self._Cn2_coeff['A3'] \
            * np.exp(
                -(h-self._Cn2_coeff['H3'])**2.
                / (2.*self._Cn2_coeff['d']**2.)
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
            if self.alts[i] == 0.:
                start = 0
            else:
                start = np.log10(self.alts[i])
            if i < n_alts-1:
                end = np.log10(self.alts[i+1])
            else:
                # Infinit altitude
                end = np.log10(_BIG_NUMBER)

            alts = np.logspace(start, end, _N_SAMPLE)
            Cn2_alts = self._get_Cn2(alts)
            Cn2_dh.append(simpson(Cn2_alts, x=alts))

        self.Cn2_dh = np.array(Cn2_dh)

    def _get_r0(self, lam, theta=0.):
        """Get r0

        Compute Fried Parameter.
        https://arxiv.org/abs/astro-ph/0202137

        Parameters
        ----------
        lam: float
            Wavelenght (in m).
        theta: float
            Zenith anlge (in deg).
            Angle between vertical and pointing.
            [Default: 0.]

        Return
        ------
        r0: float
            Fried parameter.

        """

        k = 2.*np.pi/lam
        r0_vert = (0.423 * k**2. * self.Cn2_dh)**(-3/5)

        return np.cos(theta*np.pi/180.)**(3/5.) * r0_vert

    def _get_target_seeing(self, target_seeing):

        def chi2(r0_factor):
            L0, r0_500 = self._get_L0_r0_eff(r0_factor=r0_factor)
            model = self.make_VonKarman(L0=L0, r0_500=r0_500)
            model_seeing = model.calculateFWHM()
            return (target_seeing - model_seeing)**2
        
        res = minimize(chi2, 1.)
        print(res)

    def _get_L0_r0_eff(self, r0_factor=1.):
        """Get L0 and r0 effective

        Compute the effective outer scale L0 and Fried parameter at 500nm
        r0_500.

        Parameters
        ----------
        r0_factor : float, optional
            Multiplicative factor to apply to r0 to match a taget seeing, by
            default 1.

        Returns
        -------
        tuple
            L0 and r0 effective.
        """

        L0_eff = (
            simpson(self.L0**(5/3)*self.Cn2_dh, x=self.alts) /
            simpson(self.Cn2_dh, x=self.alts)
        )**(3/5)

        r0_500_eff = (
            sum(r**(-5./3) for r in self._get_r0(500*1e-9)*r0_factor)
        )**(-3./5)

        return L0_eff, r0_500_eff

    def _get_wind(self, theta=0.):
        """Get wind speed

        Wind profile from Greenwood model.
        Wind direction is set randomly for each layers in [0., 360.].
        Ref: Greenwood (1977)

        Parameters
        ----------
        theta: float
            Zenith anlge (in deg).
            Angle between vertical and pointing.
            [Default: 0.]

        """

        coeff = self._config['wind']['GW_coeff']
        ground_speed = self._config['wind']['wind_speed']['ground']
        trop_speed = self._config['wind']['wind_speed']['trop']

        v_g = self._rng.uniform(*ground_speed, size=self._n_layers)
        v_t = self._rng.uniform(*trop_speed, size=self._n_layers)

        self.wind_speed = v_g \
            + v_t*np.exp(
                -(
                    (self.alts*np.cos(theta*np.pi/180.)
                        - coeff['H'])/coeff['T']
                    )**2.
                )

        self.wind_dir = self._rng.uniform(0., 360., size=self._n_layers)
        self.wind_dir = self.wind_dir * galsim.degrees

    def _get_screen_size(self):
        """Get screen size

        Compute the size of simulated screen based on telescope FOV.

        """

        FOV = self._config['telescope']['FOV']

        self._screen_size = 2.*_MAX_ALT*np.tan(FOV/2.*np.pi/180.)

    def make_atmosphere(self, **kwargs):
        """Make atmosphere

        Create galsim object `galsim.Atmosphere` from which we draw PSFs.

        """

        self.atm = galsim.Atmosphere(
            r0_500=self.r0_500*self._r0_factor,
            altitude=self.alts/1_000.,
            L0=self.L0,
            speed=self.wind_speed,
            direction=self.wind_dir,
            screen_size=self._screen_size,
            screen_scale=_SCREEN_SCALE,
            rng=self._galsim_rng,
            **kwargs,
        )

        return self.atm

    def make_VonKarman(self, L0=None, r0_500=None):
        """Make VonKarman

        Create a VonKarman PSF for the current atmosphere. This can be usefull
        For extremlly bright objects for which drawing with photon shooting
        or using fourrier otipcs on PhaseScreen PSF would be to long.
        It uses an effective r0_500: np.sum(r0_500s**(-5./3))**(-3./5)

        ..math::
            r_{0,500}^{eff} = \sum

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
        )

        return psf_VK
