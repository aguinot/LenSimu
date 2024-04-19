

import numpy as np

import galsim

from .utils import parser
from .psf.atmospheric import atmosphere
from .GalaxyMaker import GalaxyMaker

from tqdm.notebook import tqdm


class CCDMaker(object):
    """
    """

    def __init__(
        self,
        config,
        ccd_number,
        ccd_wcs,
        sky_level,
        exptime,
        mag_zp,
        g1,
        g2,
        target_seeing,
        seed,
    ):

        self.config_path = config
        self.config = self._load_config(config)
        self.seed = seed

        self.ccd_number = ccd_number
        self.ccd_wcs = ccd_wcs
        self.sky_level = sky_level
        self.exptime = exptime
        self.mag_zp = mag_zp
        self.g1 = g1
        self.g2 = g2
        self.target_seeing = target_seeing

        self._init_randoms(seed)
        print("Init PSF...")
        print("Target seeing:", target_seeing)
        self._init_psf()
        print("Init PSF done!")
        self.init_full_image()

    def _init_randoms(self, seed):
        """
        """

        # Init random
        self._seed = seed
        self._np_rng = np.random.RandomState(seed)
        self._galsim_rng = galsim.BaseDeviate(seed)

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
                " dictionary."
                )

        parser._check_config(config_dict, parser._config_atmo_template)

        return config_dict["telescope"]

    def _init_psf(self):

        self.gsparams = galsim.GSParams(maximum_fft_size=8192*2)
        self.atm = atmosphere(
            self.config_path,
            self.config["lam"],
            seed=self._seed,
            target_seeing=self.target_seeing,
            full_atm=False,
        )
        self.atm.init_PSF(
            pad_factor=4,
            gsparams=self.gsparams
        )
        # self.atm.atm.instantiate()

        # For bright object
        # We compute the PSF only once at center-ish of the mosaic
        # It should be fine because the object concern by such PSF are not used
        # in the weak lensing analysis.
        self.atm_bright = atmosphere(
            self.config_path,
            self.config["lam"],
            seed=self._seed,
            target_seeing=self.target_seeing,
            full_atm=False,
        )
        self.atm_bright.init_PSF(
            pad_factor=40,
            gsparams=self.gsparams,
        )

        self.bright_psf = self.atm_bright.makePSF(
            exptime=self.exptime,
            second_kick=True,
            theta=np.array([0., 0.])*galsim.arcmin,
            geometric_shooting=False,
            img_pos=(
                self.config["ccd_size"][0]/2.,
                self.config["data_sec"][3]
            ),
            ccd_num=13,
            full_atm=False,
            gsparams=self.gsparams,
        )
        # self.bright_psf = self.atm_bright.makePSF(
        #     exptime=self.exptime,
        #     second_kick=True,
        #     theta=np.array([0., 0.])*galsim.arcmin,
        #     geometric_shooting=False,
        #     img_pos=(
        #         self.config["ccd_size"][0]/2.,
        #         self.config["data_sec"][3]
        #     ),
        #     ccd_num=13,
        #     gsparams=self.gsparams,
        # )
        dummy_star = galsim.DeltaFunction().withFlux(1e7)
        dummy_obj = galsim.Convolve((dummy_star, self.bright_psf))
        dummy_obj.drawImage(nx=1, ny=1, wcs=self.ccd_wcs)

    def init_full_image(self):
        """
        """

        self.ccd_image = galsim.ImageI(self.config['ccd_size'][0],
                                       self.config['ccd_size'][1])

        self.ccd_image.setOrigin(1, 1)
        self.ccd_image.wcs = self.ccd_wcs
        if isinstance(self.ccd_wcs, galsim.AstropyWCS):
            world_center = self.ccd_wcs.wcs.wcs.crval
            self.fov_world_center = galsim.CelestialCoord(
                ra=world_center[0]*galsim.degrees,
                dec=world_center[1]*galsim.degrees,
            )
        elif isinstance(self.ccd_wcs, galsim.GSFitsWCS):
            self.fov_world_center = self.ccd_wcs.center

    def go(self, gal_catalog, star_catalog):
        """
        """

        self.add_objects(gal_catalog, star_catalog)
        self.finalize_full_image()

        return self.ccd_image, self.final_catalog

    def add_objects(self, gal_catalog, star_catalog):
        """
        """

        final_catalog = {
            'id': [], 'cat_id': [],
            'ra': [], 'dec': [], 'x': [], 'y': [],
            "z": [],
            'flux': [], 'mag': [],
            'hlr': [], 'q': [], 'beta': [], 'sersic_n': [],
            'intrinsic_g1': [], 'intrinsic_g2': [],
            'shear_g1': [], 'shear_g2': [],
            'psf_g1': [], 'psf_g2': [], 'psf_fwhm': [],
            'type': []
        }

        print("start gal")
        GalMaker = GalaxyMaker()
        id_n = 0
        for gal_cat in tqdm(gal_catalog, total=len(gal_catalog)):

            if (gal_cat["BA"] > 1) | (gal_cat["BA"] < 0):
                continue

            img_pos = self.ccd_wcs.toImage(
                galsim.CelestialCoord(
                    gal_cat['ra']*galsim.degrees,
                    gal_cat['dec']*galsim.degrees
                )
            )

            u, v = self.fov_world_center.project(
                galsim.CelestialCoord(
                    gal_cat['ra']*galsim.degrees,
                    gal_cat['dec']*galsim.degrees
                )
            )

            # if (
            #     (img_pos.x < self.config['data_sec'][0]) |
            #     (img_pos.x > self.config['data_sec'][1]) |
            #     (img_pos.y < self.config['data_sec'][2]) |
            #     (img_pos.y > self.config['data_sec'][3])
            # ):
            #     continue

            # Make galaxy model
            mag = self._convert_mag(
                gal_cat['r_SDSS_apparent_corr'],
                gal_cat['g_SDSS_apparent_corr']
            )
            flux = 10**((mag - self.mag_zp)/(-2.5))

            gal, sersic_n, intrinsic_g1, intrinsic_g2 = GalMaker.make_gal(
                flux,
                gal_cat['Re_arcsec'],
                gal_cat['BA'],
                gal_cat['PA_random'],
                gal_cat['shape/sersic_n'],
                self.g1,
                self.g2,
            )

            # Make PSF
            if flux < 1e7:
                if False:
                    if not self._check_in_image(img_pos):
                        continue
                    psf = self.atm.makePSF(
                        exptime=self.exptime,
                        second_kick=False,
                        theta=np.array([u, v]),
                        geometric_shooting=True,
                        img_pos=(img_pos.x, img_pos.y),
                        ccd_num=self.ccd_number,
                        gsparams=self.gsparams,
                    )
                else:
                    if not self._check_in_image(img_pos):
                        continue
                    psf = self.atm.makePSF(
                        exptime=self.exptime,
                        second_kick=True,
                        theta=np.array([u, v]),
                        geometric_shooting=False,
                        img_pos=(img_pos.x, img_pos.y),
                        ccd_num=self.ccd_number,
                        gsparams=self.gsparams,
                    )
            else:
                if not self._check_in_image(img_pos, bright_flux=True):
                    continue
                psf = self.bright_psf
                # psf = self.atm_bright.makePSF(
                #     exptime=self.exptime,
                #     second_kick=False,
                #     theta=np.array([u, v]),
                #     geometric_shooting=False,
                #     img_pos=(img_pos.x, img_pos.y),
                #     ccd_num=self.ccd_number,
                #     gsparams=self.gsparams,
                # )

            # Final obj
            obj = galsim.Convolve([gal, psf])

            seed_phot = self.seed + gal_cat["index"]
            stamp = self.draw_stamp(obj, img_pos, seed_phot)

            # psf_vign = self.draw_psf(psf, img_pos)
            psf_shape = psf.drawImage(
                wcs=self.ccd_wcs.local(img_pos),
            ).FindAdaptiveMom(use_sky_coords=True)

            # Integrate stamp in full image
            bounds = stamp.bounds & self.ccd_image.bounds
            if bounds.isDefined():
                self.ccd_image[bounds] += stamp[bounds]

            # Update output catalog
            final_catalog['id'].append(id_n)
            final_catalog['cat_id'].append(int(gal_cat["index"]))
            final_catalog['ra'].append(gal_cat['ra'])
            final_catalog['dec'].append(gal_cat['dec'])
            final_catalog['x'].append(img_pos.x)
            final_catalog['y'].append(img_pos.y)
            final_catalog['z'].append(gal_cat['zobs'])
            final_catalog['flux'].append(flux)
            final_catalog['mag'].append(mag)
            final_catalog['hlr'].append(gal_cat['Re_arcsec'])
            final_catalog['q'].append(gal_cat['BA'])
            final_catalog['beta'].append(gal_cat['PA_random'])
            final_catalog['sersic_n'].append(sersic_n)
            final_catalog['intrinsic_g1'].append(intrinsic_g1)
            final_catalog['intrinsic_g2'].append(intrinsic_g2)
            final_catalog['shear_g1'].append(self.g1)
            final_catalog['shear_g2'].append(self.g2)
            final_catalog['psf_g1'].append(psf_shape.observed_shape.g1)
            final_catalog['psf_g2'].append(psf_shape.observed_shape.g2)
            final_catalog['psf_fwhm'].append(psf_shape.moments_sigma*2.355)
            final_catalog['type'].append(1)
            id_n += 1

        print("gal finished")
        for star_cat in tqdm(star_catalog, total=len(star_catalog)):
            img_pos = self.ccd_wcs.toImage(
                galsim.CelestialCoord(
                    star_cat['ra']*galsim.degrees,
                    star_cat['dec']*galsim.degrees
                )
            )

            u, v = self.fov_world_center.project(
                galsim.CelestialCoord(
                    star_cat['ra']*galsim.degrees,
                    star_cat['dec']*galsim.degrees
                )
            )

            # if (
            #     (img_pos.x < self.config['data_sec'][0]) |
            #     (img_pos.x > self.config['data_sec'][1]) |
            #     (img_pos.y < self.config['data_sec'][2]) |
            #     (img_pos.y > self.config['data_sec'][3])
            # ):
            #     continue

            mag = star_cat['mag']
            flux = 10**((mag - self.mag_zp)/(-2.5))

            star = galsim.DeltaFunction().withFlux(flux)

            # Make PSF
            if flux < 1e7:
                if False:
                    if not self._check_in_image(img_pos):
                        continue
                    psf = self.atm.makePSF(
                        exptime=self.exptime,
                        second_kick=True,
                        theta=np.array([u, v]),
                        geometric_shooting=False,
                        img_pos=(img_pos.x, img_pos.y),
                        ccd_num=self.ccd_number,
                        gsparams=self.gsparams,
                    )
                else:
                    if not self._check_in_image(img_pos):
                        continue
                    psf = self.atm.makePSF(
                        exptime=self.exptime,
                        second_kick=True,
                        theta=np.array([u, v]),
                        geometric_shooting=False,
                        img_pos=(img_pos.x, img_pos.y),
                        ccd_num=self.ccd_number,
                        gsparams=self.gsparams,
                    )
            else:
                if not self._check_in_image(img_pos, bright_flux=True):
                    continue
                psf = self.bright_psf
                # psf = self.atm_bright.makePSF(
                #     exptime=self.exptime,
                #     second_kick=False,
                #     theta=np.array([u, v]),
                #     geometric_shooting=False,
                #     img_pos=(img_pos.x, img_pos.y),
                #     ccd_num=self.ccd_number,
                #     gsparams=self.gsparams,
                # )

            # Final obj
            obj = galsim.Convolve((star, psf))

            seed_phot = self.seed + star_cat["index"]
            stamp = self.draw_stamp(obj, img_pos, seed_phot)

            # psf_vign = self.draw_psf(psf, img_pos)
            psf_shape = psf.drawImage(
                wcs=self.ccd_wcs.local(img_pos),
            ).FindAdaptiveMom(use_sky_coords=True)

            # Integrate stamp in full image
            bounds = stamp.bounds & self.ccd_image.bounds
            if bounds.isDefined():
                self.ccd_image[bounds] += stamp[bounds]

            # Update output catalog
            final_catalog['id'].append(id_n)
            final_catalog['cat_id'].append(int(star_cat["index"]))
            final_catalog['ra'].append(star_cat['ra'])
            final_catalog['dec'].append(star_cat['dec'])
            final_catalog['x'].append(img_pos.x)
            final_catalog['y'].append(img_pos.y)
            final_catalog['z'].append(-10)
            final_catalog['flux'].append(flux)
            final_catalog['mag'].append(mag)
            final_catalog['hlr'].append(-10)
            final_catalog['q'].append(-10)
            final_catalog['beta'].append(-10)
            final_catalog['sersic_n'].append(-10)
            final_catalog['intrinsic_g1'].append(-10)
            final_catalog['intrinsic_g2'].append(-10)
            final_catalog['shear_g1'].append(-10)
            final_catalog['shear_g2'].append(-10)
            final_catalog['psf_g1'].append(psf_shape.observed_shape.g1)
            final_catalog['psf_g2'].append(psf_shape.observed_shape.g2)
            final_catalog['psf_fwhm'].append(psf_shape.moments_sigma*2.355)
            final_catalog['type'].append(0)
            id_n += 1

        print("star finished")
        self.final_catalog = final_catalog

    def finalize_full_image(self):
        """
        """

        background = self.get_background()
        self.ccd_image += background

        # Apply weight
        ccd_weight = np.load(
            self.config["focal_plane_file"]["weights"]
        )[self.ccd_number]
        self.ccd_image *= ccd_weight

    def get_background(self):

        sky_image = galsim.ImageI(
            self.config['ccd_size'][0],
            self.config['ccd_size'][1]
        )

        gain = np.load(
            self.config["gain"]
        )[self.ccd_number]

        sky_level_no_gain = self.sky_level * gain / self.ccd_wcs.pixelArea(
            sky_image.true_center
        )

        self.ccd_wcs.makeSkyImage(sky_image, sky_level_no_gain)

        poisson_noise = galsim.PoissonNoise(
            rng=self._galsim_rng
        )

        sky_image.addNoise(poisson_noise)
        sky_image /= gain

        return sky_image

    def draw_stamp(self, galsim_obj, img_pos, seed_phot):
        """
        """

        # Handling of position (cf. galsim demo11.py)
        # Guess: this is to have the stamp center at integer value,
        # ix_nominal, to have a good integration in final big image.
        # We account for intra-pixel shift as an offset, dx.
        rng_phot = galsim.UniformDeviate(seed_phot)
        x_nominal = img_pos.x + 0.5
        y_nominal = img_pos.y + 0.5
        ix_nominal = int(np.floor(x_nominal + 0.5))
        iy_nominal = int(np.floor(y_nominal + 0.5))
        dx = x_nominal - ix_nominal
        dy = y_nominal - iy_nominal
        offset = galsim.PositionD(dx, dy)

        if galsim_obj.flux <= 1e7:
            stamp = galsim_obj.drawImage(
                wcs=self.ccd_wcs.local(img_pos),
                offset=offset,
                method='phot',
                rng=rng_phot,
                nx=501,
                ny=501,
            )
        else:
            stamp = galsim_obj.drawImage(
                wcs=self.ccd_wcs.local(img_pos),
                offset=offset,
                method='fft',
                nx=1_501,
                ny=1_501
            )
            stamp.quantize()
            img_tmp = np.copy(stamp.array)
            img_tmp[img_tmp < 0] = 0
            poisson_noise = self._np_rng.poisson(lam=img_tmp) - img_tmp
            stamp += poisson_noise

        stamp.setCenter(ix_nominal, iy_nominal)

        return stamp

    def _check_in_image(self, pos, bright_flux=False):

        shift = 100
        if bright_flux:
            shift = 500

        stamp_bounds = galsim.BoundsI(
            int(pos.x)-shift,
            int(pos.x)+shift,
            int(pos.y)-shift,
            int(pos.y)+shift,
        )
        bounds = stamp_bounds & self.ccd_image.bounds
        is_in = bounds.isDefined()

        return is_in

    def _convert_mag(self, sdss_r, sdss_g):
        """
        """

        mc_r = sdss_r - 0.087*(sdss_g - sdss_r)

        return mc_r

    # def draw_psf(self, galsim_obj, img_pos):
    #     """
    #     """

    #     stamp = galsim_obj.drawImage(wcs=self.ccd_wcs.local(img_pos),
    #                                  nx=51,
    #                                  ny=51)

    #     return stamp.array
