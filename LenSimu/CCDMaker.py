import numpy as np

import galsim

from .utils import parser
from .psf.atmospheric import atmosphere
from .GalaxyMaker import GalaxyMaker

from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class CCDMaker(object):
    """ """

    def __init__(
        self,
        config,
        ccd_number,
        ccd_wcs,
        sky_level,
        exptime,
        mag_zp,
        target_seeing,
        seed,
    ):
        self.config_path = config
        self.config, self.config_simu = self._load_config(config)
        self.seed = seed

        self.ccd_number = ccd_number
        self.ccd_wcs = ccd_wcs
        self.sky_level = sky_level
        self.exptime = exptime
        self.mag_zp = mag_zp
        self.target_seeing = target_seeing

        self._init_randoms()
        self.init_full_image()
        print("Init PSF...")
        self._init_psf()
        print("Init PSF done!")

        # We set these variables for caching
        self._bkg_done = False
        self._wght_done = False

    def reset(self):
        self._init_randoms()
        self.init_full_image()

    def _init_randoms(self):
        """ """

        # Init random
        self._np_rng = np.random.RandomState(self.seed)
        self._galsim_rng = galsim.BaseDeviate(self.seed)

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

        return config_dict["telescope"], config_dict["simu"]

    def _init_psf(self, varying_psf=True, do_bright=True, full_atm=False):
        self.gsparams = galsim.GSParams(maximum_fft_size=8192 * 2)
        self.atm = atmosphere(
            self.config_path,
            self.config["lam"],
            seed=self.seed,
            target_seeing=self.target_seeing,
            full_atm=full_atm,
        )
        self.atm.init_PSF(pad_factor=4, gsparams=self.gsparams)

        if not varying_psf:
            center_ccd_world = self.ccd_wcs.toWorld(
                self.ccd_image.center - self.ccd_image.origin
            )
            u, v = self.fov_world_center.project(center_ccd_world)
            self.fixed_psf = self.atm.makePSF(
                exptime=self.exptime,
                second_kick=True,
                theta=np.array([u, v]),
                geometric_shooting=True,
                img_pos=(
                    self.config["ccd_size"][0] / 2.0,
                    self.config["data_sec"][3],
                ),
                ccd_num=self.ccd_number,
                full_atm=full_atm,
                gsparams=self.gsparams,
            )

            self._psf_vign = self.fixed_psf.drawImage(
                wcs=self.ccd_wcs.local(world_pos=center_ccd_world),
            )
            self._psf_shape = self._psf_vign.FindAdaptiveMom(
                use_sky_coords=True
            )

        # For bright object
        # We compute the PSF only once at center-ish of the mosaic
        # It should be fine because the object concern by such PSF are not used
        # in the weak lensing analysis.
        if do_bright:
            self.atm_bright = atmosphere(
                self.config_path,
                self.config["lam"],
                seed=self.seed,
                target_seeing=self.target_seeing,
                full_atm=full_atm,
            )
            self.atm_bright.init_PSF(
                pad_factor=self.config_simu["bright_star_pad_factor"],
                gsparams=self.gsparams,
            )

            self.bright_psf = self.atm_bright.makePSF(
                exptime=self.exptime,
                second_kick=True,
                theta=np.array([0.0, 0.0]) * galsim.arcmin,
                geometric_shooting=True,
                img_pos=(
                    self.config["ccd_size"][0] / 2.0,
                    self.config["data_sec"][3],
                ),
                ccd_num=13,
                full_atm=full_atm,
                gsparams=self.gsparams,
            )
        else:
            self.bright_psf = self.atm.makePSF(
                exptime=self.exptime,
                second_kick=True,
                theta=np.array([0.0, 0.0]) * galsim.arcmin,
                geometric_shooting=True,
                img_pos=(
                    self.config["ccd_size"][0] / 2.0,
                    self.config["data_sec"][3],
                ),
                ccd_num=13,
                full_atm=full_atm,
                gsparams=self.gsparams,
            )

        dummy_star = galsim.DeltaFunction().withFlux(1e7)
        dummy_obj = galsim.Convolve((dummy_star, self.bright_psf))
        dummy_obj.drawImage(nx=1, ny=1, wcs=self.ccd_wcs)

    def init_full_image(self):
        """ """

        self.ccd_image = galsim.ImageI(
            self.config["ccd_size"][0], self.config["ccd_size"][1]
        )

        self.ccd_image.setOrigin(1, 1)
        self.ccd_image.wcs = self.ccd_wcs
        if isinstance(self.ccd_wcs, galsim.AstropyWCS):
            world_center = self.ccd_wcs.wcs.wcs.crval
            self.fov_world_center = galsim.CelestialCoord(
                ra=world_center[0] * galsim.degrees,
                dec=world_center[1] * galsim.degrees,
            )
        elif isinstance(self.ccd_wcs, galsim.GSFitsWCS):
            self.fov_world_center = self.ccd_wcs.center

    def go(self, g1, g2, gal_catalog, star_catalog):
        """ """

        self.g1 = g1
        self.g2 = g2

        self.add_objects(gal_catalog, star_catalog)
        self.finalize_full_image()

        catalog = self.finalize_full_catalog()

        all_images = {
            "sci": self.ccd_image,
            "weight": self._weight_image,
            "bkg": self._sky_image,
        }
        if self.config_simu["save_psf"]:
            if self.config_simu["varying_psf"]:
                all_images["psf"] = self.all_psf_vign
            else:
                all_images["psf"] = [self._psf_vign]

        return all_images, catalog

    def add_objects(self, gal_catalog, star_catalog):
        """ """

        final_catalog = {
            "id": [],
            "cat_id": [],
            "ra": [],
            "dec": [],
            "x": [],
            "y": [],
            "z": [],
            "flux": [],
            "r_mag": [],
            "J_mag": [],
            "hlr": [],
            "q": [],
            "beta": [],
            "sersic_n": [],
            "intrinsic_g1": [],
            "intrinsic_g2": [],
            "shear_g1": [],
            "shear_g2": [],
            "psf_g1": [],
            "psf_g2": [],
            "psf_fwhm": [],
            "type": [],
        }
        if self.config_simu["save_psf"]:
            self.all_psf_vign = []

        GalMaker = GalaxyMaker()
        id_n = 0
        for gal_cat in tqdm(
            gal_catalog,
            total=len(gal_catalog),
            disable=not self.config_simu["verbose"],
        ):
            if (gal_cat["BA"] > 1) | (gal_cat["BA"] < 0):
                continue

            img_pos = self.ccd_wcs.toImage(
                galsim.CelestialCoord(
                    gal_cat["ra"] * galsim.degrees,
                    gal_cat["dec"] * galsim.degrees,
                )
            )
            img_pos = galsim.PositionD(img_pos.x - 1, img_pos.y - 1)

            # Make galaxy model
            mag = self._convert_mag(
                gal_cat["r_SDSS_apparent_corr"],
                gal_cat["g_SDSS_apparent_corr"],
            )
            flux = 10 ** ((mag - self.mag_zp) / (-2.5))

            gal, sersic_n, intrinsic_g1, intrinsic_g2 = GalMaker.make_gal(
                flux,
                gal_cat["Re_arcsec"],
                gal_cat["BA"],
                gal_cat["PA_random"],
                gal_cat["shape/sersic_n"],
                self.g1,
                self.g2,
            )

            # Make PSF
            if flux < 1e7:
                if not self._check_in_image(img_pos):
                    continue
                if self.config_simu["varying_psf"]:
                    u, v = self.fov_world_center.project(
                        galsim.CelestialCoord(
                            gal_cat["ra"] * galsim.degrees,
                            gal_cat["dec"] * galsim.degrees,
                        )
                    )
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
                    psf = self.fixed_psf
                is_bright = False
            else:
                if not self._check_in_image(img_pos, bright_flux=True):
                    continue
                psf = self.bright_psf
                is_bright = True

            # Final obj
            obj = galsim.Convolve([gal, psf])

            seed_phot = self.seed + gal_cat["index"]
            stamp = self.draw_stamp(
                obj, img_pos, seed_phot, is_bright=is_bright
            )

            if self.config_simu["varying_psf"]:
                psf_vign = psf.drawImage(
                    wcs=self.ccd_wcs.local(img_pos),
                )
                psf_shape = psf_vign.FindAdaptiveMom(use_sky_coords=True)
            else:
                psf_vign = self._psf_vign
                psf_shape = self._psf_shape

            # Integrate stamp in full image
            bounds = stamp.bounds & self.ccd_image.bounds
            if bounds.isDefined():
                self.ccd_image[bounds] += stamp[bounds]

            # Update output catalog
            final_catalog["id"].append(id_n)
            final_catalog["cat_id"].append(int(gal_cat["index"]))
            final_catalog["ra"].append(gal_cat["ra"])
            final_catalog["dec"].append(gal_cat["dec"])
            final_catalog["x"].append(img_pos.x)
            final_catalog["y"].append(img_pos.y)
            final_catalog["z"].append(gal_cat["zobs"])
            final_catalog["flux"].append(flux)
            final_catalog["r_mag"].append(mag)
            final_catalog["J_mag"].append(-10)
            final_catalog["hlr"].append(gal_cat["Re_arcsec"])
            final_catalog["q"].append(gal_cat["BA"])
            final_catalog["beta"].append(gal_cat["PA_random"])
            final_catalog["sersic_n"].append(sersic_n)
            final_catalog["intrinsic_g1"].append(intrinsic_g1)
            final_catalog["intrinsic_g2"].append(intrinsic_g2)
            final_catalog["shear_g1"].append(self.g1)
            final_catalog["shear_g2"].append(self.g2)
            final_catalog["psf_g1"].append(psf_shape.observed_shape.g1)
            final_catalog["psf_g2"].append(psf_shape.observed_shape.g2)
            final_catalog["psf_fwhm"].append(psf_shape.moments_sigma * 2.355)
            final_catalog["type"].append(1)
            if self.config_simu["save_psf"]:
                self.all_psf_vign.append(psf_vign)
            id_n += 1

        for star_cat in tqdm(
            star_catalog,
            total=len(star_catalog),
            disable=not self.config_simu["verbose"],
        ):
            img_pos = self.ccd_wcs.toImage(
                galsim.CelestialCoord(
                    star_cat["ra"] * galsim.degrees,
                    star_cat["dec"] * galsim.degrees,
                )
            )
            img_pos = galsim.PositionD(img_pos.x - 1, img_pos.y - 1)

            mag = star_cat["mag"]
            flux = 10 ** ((mag - self.mag_zp) / (-2.5))

            star = galsim.DeltaFunction().withFlux(flux)

            # Make PSF
            if flux < 5e5:
                if not self._check_in_image(img_pos):
                    continue
                if self.config_simu["varying_psf"]:
                    u, v = self.fov_world_center.project(
                        galsim.CelestialCoord(
                            star_cat["ra"] * galsim.degrees,
                            star_cat["dec"] * galsim.degrees,
                        )
                    )
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
                    psf = self.fixed_psf
                is_bright = False
            else:
                if not self._check_in_image(img_pos, bright_flux=True):
                    continue
                psf = self.bright_psf
                is_bright = True

            # Final obj
            obj = galsim.Convolve((star, psf))

            seed_phot = self.seed + star_cat["index"]
            stamp = self.draw_stamp(
                obj, img_pos, seed_phot, is_bright=is_bright
            )

            if self.config_simu["varying_psf"]:
                psf_vign = psf.drawImage(
                    wcs=self.ccd_wcs.local(img_pos),
                )
                psf_shape = psf_vign.FindAdaptiveMom(use_sky_coords=True)
            else:
                psf_vign = self._psf_vign
                psf_shape = self._psf_shape

            # Integrate stamp in full image
            bounds = stamp.bounds & self.ccd_image.bounds
            if bounds.isDefined():
                self.ccd_image[bounds] += stamp[bounds]

            # Update output catalog
            final_catalog["id"].append(id_n)
            final_catalog["cat_id"].append(int(star_cat["index"]))
            final_catalog["ra"].append(star_cat["ra"])
            final_catalog["dec"].append(star_cat["dec"])
            final_catalog["x"].append(img_pos.x)
            final_catalog["y"].append(img_pos.y)
            final_catalog["z"].append(-10)
            final_catalog["flux"].append(flux)
            final_catalog["r_mag"].append(mag)
            final_catalog["J_mag"].append(star_cat["J_mag"])
            final_catalog["hlr"].append(-10)
            final_catalog["q"].append(-10)
            final_catalog["beta"].append(-10)
            final_catalog["sersic_n"].append(-10)
            final_catalog["intrinsic_g1"].append(-10)
            final_catalog["intrinsic_g2"].append(-10)
            final_catalog["shear_g1"].append(-10)
            final_catalog["shear_g2"].append(-10)
            final_catalog["psf_g1"].append(psf_shape.observed_shape.g1)
            final_catalog["psf_g2"].append(psf_shape.observed_shape.g2)
            final_catalog["psf_fwhm"].append(psf_shape.moments_sigma * 2.355)
            final_catalog["type"].append(0)
            if self.config_simu["save_psf"]:
                self.all_psf_vign.append(psf_vign)
            id_n += 1

        self.final_catalog = final_catalog

    def finalize_full_image(self):
        """ """

        # Add background
        background = self.get_background()
        self.ccd_image += background

        # Apply weight
        ccd_weight = self.get_weight()
        self.ccd_image *= ccd_weight

        # Apply saturation
        saturate = self.config["saturate"]
        self.ccd_image.array[self.ccd_image.array > saturate] = saturate

    def finalize_full_catalog(self):
        keys = list(self.final_catalog.keys())

        # Get dtype
        dtype = [
            (key, np.dtype(type(self.final_catalog[key][0])).type)
            for key in keys
        ]

        # Make array
        new_arr = np.empty(len(self.final_catalog[keys[0]]), dtype=dtype)
        for key in keys:
            new_arr[key] = self.final_catalog[key]

        return new_arr

    def get_background(self):
        if self._bkg_done:
            return self._sky_image

        sky_image = galsim.ImageF(self.ccd_image.bounds)

        gain = np.load(self.config["gain"])[self.ccd_number]

        sky_level_no_gain = (
            self.sky_level
            * gain
            / self.ccd_wcs.pixelArea(sky_image.true_center)
        )

        self.ccd_wcs.makeSkyImage(sky_image, sky_level_no_gain)

        poisson_noise = galsim.PoissonNoise(rng=self._galsim_rng)
        sky_image.addNoise(poisson_noise)

        read_noise = galsim.GaussianNoise(
            rng=self._galsim_rng, sigma=self.config["read_noise"]
        )
        sky_image.addNoise(read_noise)

        sky_image /= gain

        sky_image.quantize()

        common_bound = self.ccd_image.bounds & sky_image.bounds

        self._sky_image = sky_image[common_bound]
        self._bkg_done = True

        return self._sky_image

    def get_weight(self):
        if not self.config_simu["do_weight"]:
            self._weight_image = galsim.ImageF(self.ccd_image.bounds)
            self._weight_image.fill(1.0)
            self._wght_done = True
            return self._weight_image

        if self._wght_done:
            return self._weight_image
        weight = np.load(self.config["focal_plane_file"]["weights"])[
            self.ccd_number
        ]
        weight_image = galsim.ImageF(weight)
        weight_image.setOrigin(1, 1)

        common_bound = self.ccd_image.bounds & weight_image.bounds
        self._weight_image = weight_image[common_bound]
        self._wght_done = True

        return self._weight_image

    def draw_stamp(self, galsim_obj, img_pos, seed_phot, is_bright=False):
        """ """

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

        if not is_bright:
            stamp = galsim_obj.drawImage(
                wcs=self.ccd_wcs.local(img_pos),
                offset=offset,
                method="phot",
                rng=rng_phot,
                nx=self.config_simu["obj_stamp_size"],
                ny=self.config_simu["obj_stamp_size"],
            )
        else:
            stamp = galsim_obj.drawImage(
                wcs=self.ccd_wcs.local(img_pos),
                offset=offset,
                method="fft",
                nx=self.config_simu["bright_obj_stamp_size"],
                ny=self.config_simu["bright_obj_stamp_size"],
            )
            stamp.quantize()
            img_tmp = np.copy(stamp.array)
            img_tmp[img_tmp < 0] = 0
            poisson_noise = self._np_rng.poisson(lam=img_tmp) - img_tmp
            stamp += poisson_noise

        stamp.setCenter(ix_nominal, iy_nominal)
        stamp.shift(self.ccd_image.origin)

        return stamp

    def _check_in_image(self, pos, bright_flux=False):
        shift = 100
        if bright_flux:
            shift = 500

        stamp_bounds = galsim.BoundsI(
            int(pos.x) - shift,
            int(pos.x) + shift,
            int(pos.y) - shift,
            int(pos.y) + shift,
        ).shift(self.ccd_image.origin)
        bounds = stamp_bounds & self.ccd_image.bounds
        is_in = bounds.isDefined()

        return is_in

    def _convert_mag(self, sdss_r, sdss_g):
        """ """

        mc_r = sdss_r - 0.087 * (sdss_g - sdss_r)

        return mc_r


class CCDStampMaker(CCDMaker):
    def __init__(
        self,
        expbound,
        config,
        sky_level,
        exptime,
        mag_zp,
        target_seeing,
        seed,
    ):
        self.expbound = expbound
        self.config_path = config
        self.config, self.config_simu = self._load_config(config)
        self.seed = seed

        self.ccd_number = expbound._meta["EXTVER"]
        self.ccd_wcs = expbound.wcs
        self.sky_level = sky_level
        self.exptime = exptime
        self.mag_zp = mag_zp
        self.target_seeing = target_seeing

        self._init_randoms()
        self.init_full_image()
        self._init_psf(
            varying_psf=self.config_simu["varying_psf"],
            do_bright=self.config_simu["do_bright_psf"],
            full_atm=self.config_simu["full_atm"],
        )

        # We set these variables for caching
        self._bkg_done = False
        self._wght_done = False

    def init_full_image(self):
        """ """

        image_bounds = self.expbound.image_bounds
        self.ccd_image = galsim.ImageI(image_bounds)

        self.ccd_image.wcs = self.ccd_wcs
        if isinstance(self.ccd_wcs, galsim.AstropyWCS):
            world_center = self.ccd_wcs.wcs.wcs.crval
            self.fov_world_center = galsim.CelestialCoord(
                ra=world_center[0] * galsim.degrees,
                dec=world_center[1] * galsim.degrees,
            )
        elif isinstance(self.ccd_wcs, galsim.GSFitsWCS):
            self.fov_world_center = self.ccd_wcs.center
