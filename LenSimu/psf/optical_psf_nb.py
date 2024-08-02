"""
Optical PSF (Numba)

This file contain Numba functions used to remove the atmospheric component from
the focal plane plots to get pure optic shapes

"""

import numpy as np

import numba as nb
from math import sqrt, tanh

from NumbaMinpack import minpack_sig, lmdif


@nb.njit(fastmath=True)
def _eta2g(abseta):
    """
    Adapted from GalSim
    """
    if abseta > 1.0e-4:
        return tanh(0.5 * abseta) / abseta
    else:
        absetasq = abseta * abseta
        return 0.5 + absetasq * ((-1.0 / 24.0) + absetasq * (1.0 / 240.0))


@nb.njit(fastmath=True)
def qbeta2g(q, beta):
    """
    Adapted from GalSim
    """
    eta = -np.log(q)
    return _eta2g(eta) * eta * np.exp(2j * beta)


@nb.njit(fastmath=True)
def g2qbeta(g1, g2):
    """
    Adapted from GalSim
    """
    abs_g = np.sqrt(g1**2 + g2**2)
    q = (1 - abs_g) / (1 + abs_g)

    beta = 0.5 * np.angle(g1 + g2 * 1j)

    return q, beta


@nb.njit(fastmath=True)
def get_q_new(q_opt, sigma_opt, sigma_atm):
    sig_xx = sigma_opt / sqrt(q_opt)
    sig_yy = sigma_opt * sqrt(q_opt)

    A = sig_xx**2 + sigma_atm**2
    B = sig_yy**2 + sigma_atm**2
    C = 0

    lambda_1 = (A + B + sqrt((A + B) ** 2 - 4 * (A * B - C**2))) / 2
    lambda_2 = (A + B - sqrt((A + B) ** 2 - 4 * (A * B - C**2))) / 2

    q_new = sqrt(min(lambda_1, lambda_2) / max(lambda_1, lambda_2))

    return q_new


@nb.cfunc(minpack_sig)
@nb.njit(minpack_sig)
def chi2(x_, fvec, args_):
    # q_map, sigma_opt, sigma_atm = args
    q_opt = nb.carray(x_, (1,))
    args = nb.carray(args_, (3,))
    q_new = get_q_new(
        np.float64(q_opt[0]),
        np.float64(args[1]),
        np.float64(args[2]),
    )

    fvec[0] = (np.float64(args[0]) - q_new) ** 2


CHI2_POINTER = chi2.address


@nb.njit(
    nb.types.Tuple((nb.float64[:, :, :], nb.float64[:, :, :]))(
        nb.float64,
        nb.float64,
        nb.float64[:, :, :],
        nb.float64[:, :, :],
    ),
    fastmath=True,
)
def rescale_focal_plane(sig_atm, _sig_opt, _e1_opt_arr, _e2_opt_arr):

    n_ccd, nx, ny = _e1_opt_arr.shape

    new_e1_opt_arr = np.zeros_like(_e1_opt_arr)
    new_e2_opt_arr = np.zeros_like(_e2_opt_arr)
    for k in range(n_ccd):
        q_ccd, beta_ccd = g2qbeta(_e1_opt_arr[k], _e2_opt_arr[k])
        q_ccd = q_ccd.ravel()
        beta_ccd = beta_ccd.ravel()
        g1_tmp = np.zeros_like(q_ccd)
        g2_tmp = np.zeros_like(q_ccd)
        for i in range(nx * ny):
            xsol, fvec, success, info = lmdif(
                CHI2_POINTER,
                np.array([0.1]),
                1,
                np.array([q_ccd[i], _sig_opt, sig_atm]),
            )
            new_q = xsol[0]
            g_tmp = qbeta2g(new_q, beta_ccd[i])
            g1_tmp[i] = g_tmp.real
            g2_tmp[i] = g_tmp.imag
        new_e1_opt_arr[k] = g1_tmp.reshape((nx, ny))
        new_e2_opt_arr[k] = g2_tmp.reshape((nx, ny))

    return new_e1_opt_arr, new_e2_opt_arr
