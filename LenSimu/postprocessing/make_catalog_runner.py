"""

Based on ShapePipe commit: 20b79f3
"""

import numpy as np


def save_ngmix_data(ngmix_cat):
    """Save ngmix data

    Save the ngmix catalog into the final one.

    Parameters
    ----------
    final_cat_file : io.FITSCatalog
        Final catalog.
    ngmix_cat_path : str
        Path to ngmix catalog to save.

    """

    obj_id = np.array(ngmix_cat["noshear"]["id"])

    ngmix_n_epoch = ngmix_cat["noshear"]["n_epoch_model"]
    ngmix_mcal_flags = ngmix_cat["noshear"]["mcal_flags"]

    ngmix_mom_fail = ngmix_cat["noshear"]["moments_fail"]

    ngmix_id = ngmix_cat["noshear"]["id"]
    # max_epoch = np.max(ngmix_n_epoch)

    keys = ["1M", "1P", "2M", "2P", "NOSHEAR"]
    dtype = []
    for key in keys:
        dtype += [
            (f"NGMIX_ELL_{key}", np.float64, (2,)),
            (f"NGMIX_ELL_ERR_{key}", np.float64, (2,)),
            (f"NGMIX_T_{key}", np.float64),
            (f"NGMIX_T_ERR_{key}", np.float64),
            (f"NGMIX_Tpsf_{key}", np.float64),
            (f"NGMIX_SNR_{key}", np.float64),
            (f"NGMIX_FLUX_{key}", np.float64),
            (f"NGMIX_FLUX_ERR_{key}", np.float64),
            (f"NGMIX_FLAGS_{key}", np.float64),
            (f"NGMIX_ELL_PSFo_{key}", np.float64, (2,)),
            (f"NGMIX_T_PSFo_{key}", np.float64),
        ]
    dtype += [
        ("NGMIX_N_EPOCH", np.float64),
        ("NGMIX_MCAL_FLAGS", np.float64),
        ("NGMIX_MOM_FAIL", np.float64),
    ]
    output_arr = np.ones(len(obj_id), dtype=dtype)
    # This is not ideal because it will also set the flags to -10.
    # Need to find a better way to handle this.
    output_arr.fill(-10)

    for i, id_tmp in enumerate(obj_id):
        ind = np.where(id_tmp == ngmix_id)[0]
        if len(ind) < 1:
            continue
        for key in keys:
            output_arr["NGMIX_ELL_{}".format(key)][i][0] = ngmix_cat[
                key.lower()
            ]["g1"][ind[0]]
            output_arr["NGMIX_ELL_{}".format(key)][i][1] = ngmix_cat[
                key.lower()
            ]["g2"][ind[0]]
            output_arr["NGMIX_ELL_ERR_{}".format(key)][i][0] = ngmix_cat[
                key.lower()
            ]["g1_err"][ind[0]]
            output_arr["NGMIX_ELL_ERR_{}".format(key)][i][1] = ngmix_cat[
                key.lower()
            ]["g2_err"][ind[0]]
            output_arr["NGMIX_T_{}".format(key)][i] = ngmix_cat[key.lower()][
                "T"
            ][ind[0]]
            output_arr["NGMIX_T_ERR_{}".format(key)][i] = ngmix_cat[
                key.lower()
            ]["T_err"][ind[0]]
            output_arr["NGMIX_Tpsf_{}".format(key)][i] = ngmix_cat[
                key.lower()
            ]["Tpsf"][ind[0]]
            output_arr["NGMIX_SNR_{}".format(key)][i] = ngmix_cat[key.lower()][
                "s2n"
            ][ind[0]]
            output_arr["NGMIX_FLUX_{}".format(key)][i] = ngmix_cat[
                key.lower()
            ]["flux"][ind[0]]
            output_arr["NGMIX_FLUX_ERR_{}".format(key)][i] = ngmix_cat[
                key.lower()
            ]["flux_err"][ind[0]]
            output_arr["NGMIX_FLAGS_{}".format(key)][i] = ngmix_cat[
                key.lower()
            ]["flags"][ind[0]]

            output_arr["NGMIX_ELL_PSFo_{}".format(key)][i][0] = ngmix_cat[
                key.lower()
            ]["g1_psfo_ngmix"][ind[0]]
            output_arr["NGMIX_ELL_PSFo_{}".format(key)][i][1] = ngmix_cat[
                key.lower()
            ]["g2_psfo_ngmix"][ind[0]]
            output_arr["NGMIX_T_PSFo_{}".format(key)][i] = ngmix_cat[
                key.lower()
            ]["T_psfo_ngmix"][ind[0]]

        output_arr["NGMIX_N_EPOCH"][i] = ngmix_n_epoch[ind[0]]
        output_arr["NGMIX_MCAL_FLAGS"][i] = ngmix_mcal_flags[ind[0]]
        output_arr["NGMIX_MOM_FAIL"][i] = ngmix_mom_fail[ind[0]]

    return output_arr


def save_detect_ngmix_data(input_cat):
    obj_id = np.array(input_cat["noshear"]["id"])
    ngmix_id = input_cat["noshear"]["id"]

    dtype_detect = [
        ("RA", np.float64),
        ("DEC", np.float64),
        ("X_IMAGE", np.float64),
        ("Y_IMAGE", np.float64),
        ("KRON_RAD", np.float64),
        ("FLUX_AUTO", np.float64),
        ("FLUXERR_AUTO", np.float64),
        ("FLUX_RADIUS", np.float64),
        ("MAG_AUTO", np.float64),
        ("MAGERR_AUTO", np.float64),
        ("SNR_WIN", np.float64),
        ("FLAGS", np.float64),
        ("IMAFLAGS_ISO", np.float64),
    ]

    detect_arr = np.ones(len(obj_id), dtype=dtype_detect)

    for i, id_tmp in enumerate(obj_id):
        ind = np.where(id_tmp == ngmix_id)[0]
        if len(ind) < 1:
            continue
        detect_arr["RA"][i] = input_cat["ra"][ind[0]]
        detect_arr["DEC"][i] = input_cat["dec"][ind[0]]
        detect_arr["X_IMAGE"][i] = input_cat["x"][ind[0]]
        detect_arr["Y_IMAGE"][i] = input_cat["y"][ind[0]]
        detect_arr["KRON_RAD"][i] = input_cat["kronrad"][ind[0]]
        detect_arr["FLUX_AUTO"][i] = input_cat["flux"][ind[0]]
        detect_arr["FLUXERR_AUTO"][i] = input_cat["flux_err"][ind[0]]
        detect_arr["FLUX_RADIUS"][i] = input_cat["flux_radius"][ind[0]]
        detect_arr["MAG_AUTO"][i] = input_cat["mag"][ind[0]]
        detect_arr["MAGERR_AUTO"][i] = input_cat["mag_err"][ind[0]]
        detect_arr["SNR_WIN"][i] = input_cat["snr"][ind[0]]
        detect_arr["FLAGS"][i] = input_cat["flag"][ind[0]]
        detect_arr["IMAFLAGS_ISO"][i] = input_cat["ext_flag"][ind[0]]

    ngmix_arr = save_ngmix_data(input_cat)

    dtype_final = np.dtype(detect_arr.dtype.descr + ngmix_arr.dtype.descr)

    output_arr = np.empty(len(obj_id), dtype=dtype_final)
    for key in output_arr.dtype.names:
        if "NGMIX" in key:
            output_arr[key][:] = ngmix_arr[key]
        else:
            output_arr[key][:] = detect_arr[key]

    return output_arr
