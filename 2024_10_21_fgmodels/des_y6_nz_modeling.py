import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.optimize  # noqa: E402

# fmt: off
ZVALS = np.array([
    0.   , 0.035, 0.085, 0.135, 0.185, 0.235, 0.285, 0.335, 0.385,
    0.435, 0.485, 0.535, 0.585, 0.635, 0.685, 0.735, 0.785, 0.835,
    0.885, 0.935, 0.985, 1.035, 1.085, 1.135, 1.185, 1.235, 1.285,
    1.335, 1.385, 1.435, 1.485, 1.535, 1.585, 1.635, 1.685, 1.735,
    1.785, 1.835, 1.885, 1.935, 1.985, 2.035, 2.085, 2.135, 2.185,
    2.235, 2.285, 2.335, 2.385, 2.435, 2.485, 2.535, 2.585, 2.635,
    2.685, 2.735, 2.785, 2.835, 2.885, 2.935, 2.985], dtype=np.float64)
GMODEL_COSMOS = np.array([
    0.        , 0.08246582, 1.3601444 , 1.25066374, 2.12958718,
    2.07238116, 1.12181115, 1.74694677, 1.41628421, 0.69532465,
    0.685951  , 0.61997625, 0.51828428, 0.50239732, 0.80093222,
    0.56618465, 0.35469994, 0.54471916, 0.40586034, 0.46442509,
    0.37345465, 0.15910531, 0.1814102 , 0.21069403, 0.16936997,
    0.18921011, 0.12080918, 0.09110721, 0.09599979, 0.1051503 ,
    0.08837551, 0.05902462, 0.05996257, 0.06431205, 0.0379474 ,
    0.04820887, 0.04965505, 0.0430462 , 0.03193403, 0.04069588,
    0.04148511, 0.02938825, 0.02468001, 0.02701437, 0.02319899,
    0.02105647, 0.02256154, 0.02151391, 0.01996838, 0.01486944,
    0.02031702, 0.02035628, 0.01602842, 0.01748957, 0.01953005,
    0.01923483, 0.02052126, 0.01762587, 0.01375706, 0.01627206,
    0.01461884], dtype=np.float64)
# fmt: on
ZBIN_LOW = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])
ZBIN_HIGH = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 6.0])


def mstudt_trunc(z, scale=0.01):
    return jax.nn.sigmoid((z - scale / 2) / (scale / 100))


@jax.jit
def mstudt(z, mu, sigma):
    nu = 3
    znrm = (z - mu) / sigma
    vals = jnp.power(1 + znrm * znrm / nu, -(nu + 1) / 2)
    vals = vals * mstudt_trunc(z)
    return vals


def _safe_div(numer, denom):
    denom = jnp.where(
        (denom == 0),
        1.0,
        denom,
    )
    return jnp.where(
        (numer == 0) & (denom == 0),
        0.0,
        numer / denom,
    )


_mstudt_d1 = jax.grad(mstudt)
_mstudt_d2 = jax.grad(_mstudt_d1)
_mstudt_d3 = jax.grad(_mstudt_d2)
_mstudt_d4 = jax.grad(_mstudt_d3)


def _fmodel_mstudt4(z, a0, a1, a2, a3, a4, mu, sigma):
    val = mstudt(z, mu, sigma)
    val_d1 = _mstudt_d1(z, mu, sigma)
    val_d2 = _mstudt_d2(z, mu, sigma)
    val_d3 = _mstudt_d3(z, mu, sigma)
    val_d4 = _mstudt_d4(z, mu, sigma)

    return (
        a0
        - a1 * _safe_div(val_d1, val)
        + a2 * _safe_div(val_d2, val) / 2.0
        - a3 * _safe_div(val_d3, val) / 6.0
        + a4 * _safe_div(val_d4, val) / 24.0
    )


fmodel_mstudt4 = jax.jit(
    jax.vmap(_fmodel_mstudt4, in_axes=[0, None, None, None, None, None, None, None]),
)


@jax.jit
def mstudt_nrm(z, mu, sigma):
    vals = mstudt(z, mu, sigma)
    return vals / sompz_integral_nojit(vals, z, 0, 6)


def gmodel_template_cosmos():
    return GMODEL_COSMOS


def fit_nz_data_for_template_params(nzs):
    """Fit the modified Student's t model to the n(z) data.

    Parameters
    ----------
    nzs : dict mapping bin index to n(z).
        The input n(z) data.

    Returns
    -------
    params : dict mapping bin index to parameter tuples
        The fitted parameters for each bin.
    """
    params = {}
    popt = None
    for i in range(len(nzs)):
        popt, pcov = scipy.optimize.curve_fit(
            mstudt_nrm,
            ZVALS,
            nzs[i] / sompz_integral(nzs[i], ZVALS, 0, 6),
            p0=(1, 0.1) if popt is None else popt,
        )
        params[i] = tuple(v for v in popt)

    return params


def sompz_integral_nojit(y, x, low, high):
    """Integrate a linearly interpolated set of values
    on a grid in a range (low, high).

    Parameters
    ----------
    y : array
        The values to integrate.
    x : array
        The grid points. They must be sorted, but need not
        be evenly spaced.
    low : float
        The lower bound of the integration range.
    high : float
        The upper bound of the integration range.

    Returns
    -------
    float
        The integral of the values in the range (low, high).
    """
    low = jnp.minimum(x[-1], jnp.maximum(low, x[0]))
    high = jnp.minimum(x[-1], jnp.maximum(high, x[0]))
    low_ind = jnp.digitize(low, x)
    # for the lower index we do not use right=True, but
    # we still want to clip to a valid index of x
    low_ind = jnp.minimum(low_ind, x.shape[0] - 1)
    high_ind = jnp.digitize(high, x, right=True)
    dx = x[1:] - x[:-1]

    # high point not in same bin as low point
    not_in_single_bin = high_ind > low_ind

    # fractional bit on the left
    ileft = jax.lax.select(
        not_in_single_bin,
        (y[low_ind - 1] + y[low_ind])
        / 2.0
        * (1.0 - (low - x[low_ind - 1]) / dx[low_ind - 1])
        * dx[low_ind - 1],
        (y[low_ind - 1] + y[low_ind]) / 2.0 * (high - low),
    )

    # fractional bit on the right
    iright = jax.lax.select(
        not_in_single_bin,
        (y[high_ind - 1] + y[high_ind]) / 2.0 * (high - x[high_ind - 1]),
        0.0,
    )

    # central bits
    yint = (y[1:] + y[:-1]) / 2.0 * dx
    yind = jnp.arange(yint.shape[0])
    msk = (yind >= low_ind) & (yind < high_ind - 1)
    icen = jax.lax.select(
        jnp.any(msk),
        jnp.sum(
            jnp.where(
                msk,
                yint,
                jnp.zeros_like(yint),
            )
        ),
        0.0,
    )

    return ileft + icen + iright


sompz_integral = jax.jit(sompz_integral_nojit)
