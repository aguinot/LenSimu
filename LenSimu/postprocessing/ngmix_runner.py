"""

Based on ShapePipe commit: 20b79f3
"""

# import re
from functools import partial

import numpy as np
from scipy.stats import median_abs_deviation as mad

import ngmix
from ngmix.observation import Observation, ObsList

import galsim

sigma_mad = partial(mad, scale="normal")


def get_prior(rng, pixel_scale):
    """Get Prior.

    Return prior for the different parameters.

    Returns
    -------
    ngmix.priors
        Priors for the different parameters

    """
    # Prior on ellipticity. Details do not matter, as long
    # as it regularizes the fit. From Bernstein & Armstrong 2014
    g_sigma = 0.4
    g_prior = ngmix.priors.GPriorBA(g_sigma, rng=rng)

    # 2-d Gaussian prior on the center row and column center
    # (relative to the center of the jacobian, which
    # would be zero) and the sigma of the Gaussians.
    # Units same as jacobian, probably arcsec
    row, col = 0.0, 0.0
    row_sigma, col_sigma = pixel_scale, pixel_scale
    cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma, rng)

    # Size prior. Instead of flat, two-sided error function (TwoSidedErf)
    # could be used
    Tminval = -10.0  # arcsec squared
    Tmaxval = 1.0e6
    T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

    # Flux prior. Bounds need to make sense for
    # images in question
    Fminval = -1.0e4
    Fmaxval = 1.0e9
    F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval, rng=rng)

    # Joint prior, combine all individual priors
    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior, g_prior, T_prior, F_prior
    )

    return prior


def get_guess(
    img,
    pixel_scale,
    guess_flux_unit="img",
    guess_size_type="T",
    guess_size_unit="sky",
    guess_centroid=True,
    guess_centroid_unit="sky",
):
    r"""Get Guess.

    Get the guess vector for the NGMIX shape measurement
    ``[center_x, center_y, g1, g2, size_T, flux]``.
    No guesses are given for the ellipticity ``(0, 0)``.

    Parameters
    ----------
    img : numpy.ndarray
        Array containing the image
    pixel_scale : float
        Approximation of the pixel scale
    guess_flux_unit : str
        If ``img`` returns the flux in pixel units, otherwise if ``sky``
        returns the flux in :math:`{\rm arcsec}^{-2}`
    guess_size_type : str
        If ``T`` returns the size in quadrupole moments definition
        :math:`2\sigma^2`, otherwise if ``sigma`` returns the moments
        :math:`\sigma`
    guess_size_unit : str
        If ``img`` returns the size in pixel units, otherwise if ``sky``
        returns the size in arcsec
    guess_centroid : bool
        If ``True``, will return a guess on the object centroid, otherwise if
        ``False``, will return the image centre
    guess_centroid_unit : str
        If ``img`` returns the centroid in pixel unit, otherwise if ``sky``
        returns the centroid in arcsec

    Returns
    -------
    numpy.ndarray
        Return the guess array ``[center_x, center_y, g1, g2, size_T, flux]``

    Raises
    ------
    GalSimHSMError
        For an error in the computation of adaptive moments
    ValueError
        For invalid unit guess types

    """
    galsim_img = galsim.Image(img, scale=pixel_scale)

    hsm_shape = galsim.hsm.FindAdaptiveMom(galsim_img, strict=False)

    error_msg = hsm_shape.error_message

    if error_msg != "":
        raise galsim.hsm.GalSimHSMError(
            f"Error in adaptive moments :\n{error_msg}"
        )

    if guess_flux_unit == "img":
        guess_flux = hsm_shape.moments_amp
    elif guess_flux_unit == "sky":
        guess_flux = hsm_shape.moments_amp / pixel_scale**2
    else:
        raise ValueError(
            f"invalid guess_flux_unit '{guess_flux_unit}',"
            + " must be one of 'img', 'sky'"
        )

    if guess_size_unit == "img":
        size_unit = 1.0
    elif guess_size_unit == "sky":
        size_unit = pixel_scale
    else:
        raise ValueError(
            "invalid guess_size_unit '{guess_size_unit}',"
            + "must be one of 'img', 'sky'"
        )

    if guess_size_type == "sigma":
        guess_size = hsm_shape.moments_sigma * size_unit
    elif guess_size_type == "T":
        guess_size = 2 * (hsm_shape.moments_sigma * size_unit) ** 2

    if guess_centroid_unit == "img":
        centroid_unit = 1
    elif guess_centroid_unit == "sky":
        centroid_unit = pixel_scale
    else:
        raise ValueError(
            f"invalid guess_centroid_unit '{guess_centroid_unit}',"
            + "  must be one of 'img', 'sky'"
        )

    if guess_centroid:
        guess_centroid = (
            hsm_shape.moments_centroid - galsim_img.center
        ) * centroid_unit
    else:
        guess_centroid = galsim_img.center * centroid_unit

    guess = np.array(
        [guess_centroid.x, guess_centroid.y, 0.0, 0.0, guess_size, guess_flux]
    )

    return guess


def get_jacob(wcs, ra, dec):
    """Get jacobian

    Return the jacobian of the wcs at the required position.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS object for wich we want the jacobian.
    ra : float
        Ra position of the center of the vignet (in Deg).
    dec : float
        Dec position of the center of the vignet (in Deg).

    Returns
    -------
    galsim_jacob : galsim.wcs.BaseWCS.jacobian
        Jacobian of the WCS at the required position.

    """

    g_wcs = galsim.fitswcs.AstropyWCS(wcs=wcs)
    world_pos = galsim.CelestialCoord(
        ra=ra * galsim.angle.degrees, dec=dec * galsim.angle.degrees
    )
    galsim_jacob = g_wcs.jacobian(world_pos=world_pos)

    return galsim_jacob


def get_noise(gal, weight, guess, pixel_scale, thresh=1.2):
    r"""Get Noise.

    Compute the sigma of the noise from an object postage stamp.
    Use a guess on the object size, ellipticity and flux to create a window
    function.

    Parameters
    ----------
    gal : numpy.ndarray
        Galaxy image
    weight : numpy.ndarray
        Weight image
    guess : list
        Gaussian parameters fot the window function
        ``[x0, y0, g1, g2, T, flux]``
    pixel_scale : float
        Pixel scale of the galaxy image
    thresh : float, optional
        Threshold to cut the window function,
        cut = ``thresh`` * :math:`\sigma_{\rm noise}`;  the default is ``1.2``

    Returns
    -------
    float
        Sigma of the noise on the galaxy image

    """
    img_shape = gal.shape

    m_weight = weight != 0

    sig_tmp = sigma_mad(gal[m_weight])

    gauss_win = galsim.Gaussian(sigma=np.sqrt(guess[4] / 2), flux=guess[5])
    gauss_win = gauss_win.shear(g1=guess[2], g2=guess[3])
    gauss_win = gauss_win.drawImage(
        nx=img_shape[0], ny=img_shape[1], scale=pixel_scale
    ).array

    m_weight = weight[gauss_win < thresh * sig_tmp] != 0

    sig_noise = sigma_mad(gal[gauss_win < thresh * sig_tmp][m_weight])

    return sig_noise


def do_ngmix_metacal(
    gals,
    psfs,
    psfs_sigma,
    weights,
    flags,
    jacob_list,
    prior,
    pixel_scale,
    rng,
):
    """Do Ngmix Metacal.

    Perform the metacalibration on a multi-epoch object and return the joint
    shape measurement with NGMIX.

    Parameters
    ----------
    gals : list
        List of the galaxy vignets
    psfs : list
        List of the PSF vignets
    psfs_sigma : list
        List of the sigma PSFs
    weights : list
        List of the weight vignets
    flags : list
        List of the flag vignets
    jacob_list : list
        List of the Jacobians
    prior : ngmix.priors
        Priors for the fitting parameters
    pixel_scale : float
        pixel scale in arcsec

    Returns
    -------
    dict
        Dictionary containing the results of NGMIX metacal

    """
    n_epoch = len(gals)

    if n_epoch == 0:
        raise ValueError("0 epoch to process")

    # psfs_sigma /= pixel_scale
    Tguess = np.mean(2.0 * (psfs_sigma * pixel_scale) ** 2)

    # Setup PSF runner (pre metacal)
    psf_fitter = ngmix.admom.AdmomFitter(rng=rng)
    psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1)
    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=5,
    )

    # Setup Galaxy runner
    fitter = ngmix.fitting.Fitter(model="gauss", prior=prior)
    # make parameter guesses based on a psf flux and a rough T
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng,
        T=Tguess,
        prior=prior,
    )
    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=5,
    )
    # Setup PSF runner (post metacal)
    # We fit a gaussian so 1 should be enough
    psf_ngauss = 1
    psf_fitter2 = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss)
    psf_guesser2 = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)
    psf_runner2 = ngmix.runners.PSFRunner(
        fitter=psf_fitter2, guesser=psf_guesser2, ntry=5
    )
    boot = ngmix.bootstrap.Bootstrapper(runner=runner, psf_runner=psf_runner2)

    # Make observation
    gal_obs_list = ObsList()
    # T_guess_psf = []
    psf_res_gT = {
        "g_PSFo": np.array([0.0, 0.0]),
        "g_err_PSFo": np.array([0.0, 0.0]),
        "T_PSFo": 0.0,
        "T_err_PSFo": 0.0,
    }
    gal_guess = []
    gal_guess_flag = True
    wsum = 0
    for n_e in range(n_epoch):
        psf_jacob = ngmix.Jacobian(
            row=(psfs[0].shape[0] - 1) / 2,
            col=(psfs[0].shape[1] - 1) / 2,
            wcs=jacob_list[n_e],
        )

        psf_obs = Observation(
            psfs[n_e],
            weight=np.ones_like(psfs[n_e]) * 1.0 / (1e-5) ** 2.0,
            jacobian=psf_jacob,
        )

        weight_map = np.copy(weights[n_e])
        weight_map[np.where(flags[n_e] != 0)] = 0.0
        weight_map[weight_map != 0] = 1

        # psf_guess = np.array([0., 0., 0., 0., psf_T, 1.])
        try:
            psf_res = psf_runner.go(psf_obs)
        except Exception:
            continue

        # Gal guess
        try:
            gal_guess_tmp = get_guess(
                gals[n_e], pixel_scale, guess_size_type="sigma"
            )
        except Exception:
            gal_guess_flag = False
            gal_guess_tmp = np.array([0.0, 0.0, 0.0, 0.0, 1, 100])

        # Recenter jacobian if necessary
        gal_jacob = ngmix.Jacobian(
            row=(gals[0].shape[0] - 1) / 2,  # + gal_guess_tmp[0],
            col=(gals[0].shape[1] - 1) / 2,  # + gal_guess_tmp[1],
            wcs=jacob_list[n_e],
        )

        # Noise handling
        if gal_guess_flag:
            sig_noise = get_noise(
                gals[n_e],
                weight_map,
                gal_guess_tmp,
                pixel_scale,
            )
        else:
            sig_noise = sigma_mad(gals[n_e])

        noise_img = rng.randn(*gals[n_e].shape) * sig_noise
        noise_img_gal = rng.randn(*gals[n_e].shape) * sig_noise

        gal_masked = np.copy(gals[n_e])
        if len(np.where(weight_map == 0)[0]) != 0:
            gal_masked[weight_map == 0] = noise_img_gal[weight_map == 0]

        weight_map *= 1 / sig_noise**2

        # Original PSF fit
        w_tmp = np.sum(weight_map)
        psf_res["g"] = np.array(ngmix.shape.e1e2_to_g1g2(*psf_res["e"]))
        psf_res_gT["g_PSFo"] += psf_res["g"] * w_tmp
        psf_res_gT["g_err_PSFo"] += (
            np.array(
                [
                    -1,
                    -1,
                ]
            )
            * w_tmp
        )
        psf_res_gT["T_PSFo"] += psf_res["T"] * w_tmp
        psf_res_gT["T_err_PSFo"] += psf_res["T_err"] * w_tmp
        wsum += w_tmp

        gal_obs = Observation(
            gal_masked,
            weight=weight_map,
            jacobian=gal_jacob,
            psf=psf_obs,
            noise=noise_img,
        )

        if gal_guess_flag:
            gal_guess_tmp[:2] = 0
            gal_guess.append(gal_guess_tmp)

        gal_obs_list.append(gal_obs)
        # T_guess_psf.append(psf_T)
        gal_guess_flag = True

    if wsum == 0:
        raise ZeroDivisionError("Sum of weights = 0, division by zero")

    # Normalize PSF fit output
    for key in psf_res_gT.keys():
        psf_res_gT[key] /= wsum

    # Gal guess handling
    fail_get_guess = False
    if len(gal_guess) == 0:
        fail_get_guess = True

    # metacal specific parameters
    metacal_pars = {
        "types": ["noshear", "1p", "1m", "2p", "2m"],
        "step": 0.01,
        "psf": "gauss",
        "fixnoise": True,
        "use_noise_image": True,
    }

    obs_dict_mcal = ngmix.metacal.get_all_metacal(
        gal_obs_list, rng=rng, **metacal_pars
    )
    res = {"mcal_flags": 0}

    for key in sorted(obs_dict_mcal):
        # fres = runner.go(obs_dict_mcal[key])
        fres = boot.go(obs_dict_mcal[key])

        res["mcal_flags"] |= fres["flags"]
        tres = {}

        for name in fres.keys():
            tres[name] = fres[name]
        tres["flags"] = fres["flags"]

        wsum = 0
        Tpsf_sum = 0
        gpsf_sum = np.zeros(2)
        npsf = 0
        for obs in obs_dict_mcal[key]:
            # if hasattr(obs, 'psf_nopix'):
            #     try:
            #         psf_res = psf_runner.go(obs.psf_nopix)
            #     except Exception:
            #         continue
            #     psf_res['g'] = np.array(
            #         ngmix.shape.e1e2_to_g1g2(*psf_res['e'])
            #     )
            # else:
            #     try:
            #         psf_res = psf_runner.go(obs.psf)
            #     except Exception:
            #         continue
            #     psf_res['g'] = np.array(
            #         ngmix.shape.e1e2_to_g1g2(*psf_res['e'])
            #     )
            psf_res = obs.psf.meta["result"]
            g1, g2 = psf_res["g"]
            T = psf_res["T"]

            # TODO we sometimes use other weights
            twsum = obs.weight.sum()

            wsum += twsum
            gpsf_sum[0] += g1 * twsum
            gpsf_sum[1] += g2 * twsum
            Tpsf_sum += T * twsum
            npsf += 1

        tres["gpsf"] = gpsf_sum / wsum
        tres["Tpsf"] = Tpsf_sum / wsum

        res[key] = tres
        res[key]["ntry"] = -10
    # result dictionary, keyed by the types in metacal_pars above
    metacal_res = res

    metacal_res.update(psf_res_gT)
    metacal_res["moments_fail"] = fail_get_guess

    return metacal_res


def compile_results(results):
    """Compile results

    Prepare the results of ngmix before saving.

    Parameters
    ----------
    results : dict
        Dictionary containing the results of ngmix metacal.

    Returns
    -------
    output_dict : dict
        Dictionary containing ready to be saved.

    """

    names = ["1m", "1p", "2m", "2p", "noshear"]
    names2 = [
        "id",
        "n_epoch_model",
        "moments_fail",
        "ntry_fit",
        "g1_psfo_ngmix",
        "g2_psfo_ngmix",
        "T_psfo_ngmix",
        "g1_err_psfo_ngmix",
        "g2_err_psfo_ngmix",
        "T_err_psfo_ngmix",
        "g1",
        "g1_err",
        "g2",
        "g2_err",
        "T",
        "T_err",
        "Tpsf",
        "g1_psf",
        "g2_psf",
        "flux",
        "flux_err",
        "s2n",
        "flags",
        "mcal_flags",
    ]
    output_dict = {k: {kk: [] for kk in names2} for k in names}
    for i in range(len(results)):
        for name in names:
            output_dict[name]["id"].append(results[i]["obj_id"])
            output_dict[name]["n_epoch_model"].append(
                results[i]["n_epoch_model"]
            )
            output_dict[name]["moments_fail"].append(
                results[i]["moments_fail"]
            )
            output_dict[name]["ntry_fit"].append(results[i][name]["ntry"])
            output_dict[name]["g1_psfo_ngmix"].append(results[i]["g_PSFo"][0])
            output_dict[name]["g2_psfo_ngmix"].append(results[i]["g_PSFo"][1])
            output_dict[name]["g1_err_psfo_ngmix"].append(
                results[i]["g_err_PSFo"][0]
            )
            output_dict[name]["g2_err_psfo_ngmix"].append(
                results[i]["g_err_PSFo"][1]
            )
            output_dict[name]["T_psfo_ngmix"].append(results[i]["T_PSFo"])
            output_dict[name]["T_err_psfo_ngmix"].append(
                results[i]["T_err_PSFo"]
            )
            output_dict[name]["g1"].append(results[i][name]["g"][0])
            output_dict[name]["g1_err"].append(results[i][name]["pars_err"][2])
            output_dict[name]["g2"].append(results[i][name]["g"][1])
            output_dict[name]["g2_err"].append(results[i][name]["pars_err"][3])
            output_dict[name]["T"].append(results[i][name]["T"])
            output_dict[name]["T_err"].append(results[i][name]["T_err"])
            output_dict[name]["Tpsf"].append(results[i][name]["Tpsf"])
            output_dict[name]["g1_psf"].append(results[i][name]["gpsf"][0])
            output_dict[name]["g2_psf"].append(results[i][name]["gpsf"][1])
            output_dict[name]["flux"].append(results[i][name]["flux"])
            output_dict[name]["flux_err"].append(results[i][name]["flux_err"])

            try:
                output_dict[name]["s2n"].append(results[i][name]["s2n"])
            except Exception:
                output_dict[name]["s2n"].append(results[i][name]["s2n_r"])
            output_dict[name]["flags"].append(results[i][name]["flags"])
            output_dict[name]["mcal_flags"].append(results[i]["mcal_flags"])

    return output_dict


# def process(
#     tile_cat_path,
#     gal_vignet_path,
#     bkg_vignet_path,
#     psf_vignet_path,
#     weight_vignet_path,
#     flag_vignet_path,
#     f_wcs_path,
#     w_log,
#     id_obj_min=-1,
#     id_obj_max=-1,
# ):
#     """Process

#     Process function.

#     Parameters
#     ----------
#     tile_cat_path: str
#         Path to the tile SExtractor catalog.
#     gal_vignet_path: str
#         Path to the galaxy vignets catalog.
#     bkg_vignet_path: str
#         Path to the background vignets catalog.
#     psf_vignet_path: str
#         Path to the PSF vignets catalog.
#     weight_vignet_path: str
#         Path to the weight vignets catalog.
#     flag_vignet_path: str
#         Path to the flag vignets catalog.
#     f_wcs_path: str
#         Path to the log file containing the WCS for each CCDs.
#     w_log: log file object
#         log file
#     id_obj_min: int, optional, default=-1
#         minimum object ID to be processed if > 0
#     id_obj_max: int, optional, default=-1
#         maximum object ID to be processed if > 0

#     Returns
#     -------
#     final_res: dict
#         Dictionary containing the ngmix metacal results.

#     """

#     tile_cat = io.FITSCatalog(tile_cat_path, SEx_catalog=True)
#     tile_cat.open()
#     obj_id = np.copy(tile_cat.get_data()["NUMBER"])
#     tile_vign = np.copy(tile_cat.get_data()["VIGNET"])
#     tile_ra = np.copy(tile_cat.get_data()["XWIN_WORLD"])
#     tile_dec = np.copy(tile_cat.get_data()["YWIN_WORLD"])
#     tile_cat.close()

#     f_wcs_file = SqliteDict(f_wcs_path)
#     gal_vign_cat = SqliteDict(gal_vignet_path)
#     bkg_vign_cat = SqliteDict(bkg_vignet_path)
#     psf_vign_cat = SqliteDict(psf_vignet_path)
#     weight_vign_cat = SqliteDict(weight_vignet_path)
#     flag_vign_cat = SqliteDict(flag_vignet_path)

#     final_res = []
#     prior = get_prior()

#     count = 0
#     id_first = -1
#     id_last = -1

#     for i_tile, id_tmp in enumerate(obj_id):
#         if id_obj_min > 0 and id_tmp < id_obj_min:
#             continue
#         if id_obj_max > 0 and id_tmp > id_obj_max:
#             continue

#         if id_first == -1:
#             id_first = id_tmp
#         id_last = id_tmp

#         count = count + 1

#         gal_vign = []
#         psf_vign = []
#         sigma_psf = []
#         weight_vign = []
#         flag_vign = []
#         jacob_list = []
#         if (psf_vign_cat[str(id_tmp)] == "empty") or (
#             gal_vign_cat[str(id_tmp)] == "empty"
#         ):
#             continue
#         psf_expccd_name = list(psf_vign_cat[str(id_tmp)].keys())
#         for expccd_name_tmp in psf_expccd_name:
#             gal_vign_tmp = \
#                 gal_vign_cat[str(id_tmp)][expccd_name_tmp]["VIGNET"]
#             if len(np.where(gal_vign_tmp.ravel() == 0)[0]) != 0:
#                 continue

#             bkg_vign_tmp = \
#                 bkg_vign_cat[str(id_tmp)][expccd_name_tmp]["VIGNET"]
#             gal_vign_sub_bkg = gal_vign_tmp - bkg_vign_tmp

#             tile_vign_tmp = np.copy(tile_vign[i_tile])
#             flag_vign_tmp = flag_vign_cat[str(id_tmp)][expccd_name_tmp][
#                 "VIGNET"
#             ]
#             flag_vign_tmp[np.where(tile_vign_tmp == -1e30)] = 2**10
#             v_flag_tmp = flag_vign_tmp.ravel()
#             if len(np.where(v_flag_tmp != 0)[0]) / (51 * 51) > 1 / 3.0:
#                 continue

#             exp_name, ccd_n = re.split("-", expccd_name_tmp)
#             jacob_tmp = get_jacob(
#                 f_wcs_file[exp_name][int(ccd_n)],
#                 tile_ra[i_tile],
#                 tile_dec[i_tile],
#             )

#             gal_vign.append(gal_vign_sub_bkg)
#             psf_vign.append(
#                 psf_vign_cat[str(id_tmp)][expccd_name_tmp]["VIGNET"]
#             )
#             sigma_psf.append(
#                 psf_vign_cat[str(id_tmp)][expccd_name_tmp]["SHAPES"][
#                     "SIGMA_PSF_HSM"
#                 ]
#             )
#             weight_vign.append(
#                 weight_vign_cat[str(id_tmp)][expccd_name_tmp]["VIGNET"]
#             )
#             flag_vign.append(flag_vign_tmp)
#             jacob_list.append(jacob_tmp)

#         if len(gal_vign) == 0:
#             continue
#         try:
#             res = do_ngmix_metacal(
#                 gal_vign,
#                 psf_vign,
#                 sigma_psf,
#                 weight_vign,
#                 flag_vign,
#                 jacob_list,
#                 prior,
#             )
#         except Exception as ee:
#             w_log.info(
#                 "ngmix failed for object ID={}.\nMessage: {}".format(
#                     id_tmp, ee
#                 )
#             )
#             continue

#         res["obj_id"] = id_tmp
#         res["n_epoch_model"] = len(gal_vign)
#         final_res.append(res)

#     w_log.info(
#         "ngmix loop over objects finished, processed {} "
#         "objects, id first/last={}/{}".format(count, id_first, id_last)
#     )

#     f_wcs_file.close()
#     gal_vign_cat.close()
#     bkg_vign_cat.close()
#     flag_vign_cat.close()
#     weight_vign_cat.close()
#     psf_vign_cat.close()

#     return final_res
