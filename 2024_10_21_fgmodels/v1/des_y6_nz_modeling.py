from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import ultraplot as uplt  # noqa: E402

# made in notebook des-y6-nz-fits-gmodel-tests.ipynb
# fmt: off
GMODEL_COSMOS_NZ = np.array(
    [
        0.07959297, 1.3127611 , 1.20709441, 2.05539883, 2.0001857 ,
        1.08273066, 1.68608845, 1.36694518, 0.67110165, 0.66205455,
        0.59837816, 0.50022884, 0.48489532, 0.77303018, 0.5464605 ,
        0.34234327, 0.5257428 , 0.3917214 , 0.44824593, 0.36044462,
        0.15356257, 0.17509042, 0.20335409, 0.16346964, 0.18261861,
        0.11660055, 0.08793331, 0.09265545, 0.10148718, 0.08529677,
        0.05696838, 0.05787366, 0.06207161, 0.03662543, 0.04652942,
        0.04792522, 0.0415466 , 0.03082155, 0.03927816, 0.04003989,
        0.02836445, 0.02382023, 0.02607327, 0.02239081, 0.02032293,
        0.02177557, 0.02076443, 0.01927274, 0.01435144, 0.01960923,
        0.01964712, 0.01547004, 0.01688029, 0.01884968, 0.01856474,
        0.01980636, 0.01701184, 0.01327781, 0.01570519, 0.01410957,
        0.01191486, 0.00897651, 0.01219978, 0.01082054, 0.00800058,
        0.00952325, 0.00941157, 0.0093954 , 0.00969362, 0.00960027,
        0.01553798, 0.00565781, 0.52428815, 0.003932  , 0.00323953,
        0.00355996, 0.00335961, 0.0039129 , 0.00344086, 0.03027401
    ],
    dtype=np.float64,
)
# fmt: on
DZ = 0.05
Z0 = 0.035
ZBIN_LOW = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])
ZBIN_HIGH = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 6.0])


def nz_binned_to_interp(nz, dz, z0):
    """Convert the binned n(z) to the linearly interpolated n(z).

    The total integral value of the n(z) (i.e., `np.sum(nz)`)
    is preserved.

    Parameters
    ----------
    nz : array
        The binned n(z) values (i.e., each value is the integral
        of the underlying n(z) over the bin from -dz/2 to +dz/2
        about the center of the bin).
    dz : float
        The bin width.
    z0 : float
        The center of the first bin.

    Returns
    -------
    z : array, shape (nz.size + 2,)
        The redshift values for the linearly interpolated dndz.
        The two extra values are at the ends of the interpolation
        for the first and last bins.
    dndz : array, shape (nz.size + 2,)
        The linearly interpolated dn(z)/dz values. The first and
        last values will be zero.
    """
    # the first bin's interpolation kernel is truncated to end at zero
    # if it goes below zero
    # an untruncated kernel goes from -dz to dz about each bin's center
    first_zval = jnp.maximum(z0 - dz, 0.0)
    fbin_dist_left_to_peak = (
        z0 - first_zval
        # ^ this factor is the location of the first bin's left most influence
        # if it is less than zero, we truncate
    )
    dndz = jnp.concatenate(
        [
            jnp.zeros(1),
            jnp.array([nz[0] / ((fbin_dist_left_to_peak + dz) / 2)]),
            nz[1:] / dz,
            jnp.zeros(1),
        ]
    )
    z = jnp.concatenate(
        [
            jnp.ones(1) * first_zval,
            jnp.arange(nz.shape[0] + 1) * dz + z0,
        ]
    )
    return z, dndz


GMODEL_COSMOS_Z, GMODEL_COSMOS_DNDZ = nz_binned_to_interp(GMODEL_COSMOS_NZ, DZ, Z0)


@jax.jit
def gmodel_template_cosmos(z):
    return jnp.interp(z, GMODEL_COSMOS_Z, GMODEL_COSMOS_DNDZ, left=0.0, right=0.0)


@jax.jit
def compute_lin_interp_mean(z, dndz):
    """Compute the mean of a linearly interpolated function.

    Parameters
    ----------
    z : array
        The grid points of the linearly interpolated function.
    dndz : array
        The values of the linearly interpolated function at the grid points.

    Returns
    -------
    float
        The mean of the function.
    """
    x1 = z[1:]
    x0 = z[:-1]
    y1 = dndz[1:]
    y0 = dndz[:-1]
    numer = jnp.sum(1 / 6 * (x1 - x0) * (x0 * (2 * y0 + y1) + x1 * (y0 + 2 * y1)))
    denom = jnp.sum((y0 + y1) * (x1 - x0) / 2)
    return numer / denom


@partial(jax.jit, static_argnames=("dz", "z0"))
def compute_nz_binned_mean(nz, dz=DZ, z0=Z0):
    """Compute the mean redshift of a binned n(z).

    Parameters
    ----------
    nz : array
        The binned n(z) values (i.e., each value is the integral
        of the underlying n(z) over the bin from -dz/2 to +dz/2
        about the center of the bin).
    dz : float, optional
        The bin width.
    z0 : float, optional
        The center of the first bin.

    Returns
    -------
    float
        The mean redshift of the binned n(z).
    """
    z, dndz = nz_binned_to_interp(nz, dz, z0)
    return compute_lin_interp_mean(z, dndz)


def sompz_integral_nojit(nz, low, high, dz=DZ, z0=Z0):
    """Integrate a binned n(z) over a range transforming it to
    a linearly interpolated n(z) in the process.

    Parameters
    ----------
    nz : array
        The binned n(z) values (i.e., each value is the integral
        of the underlying n(z) over the bin from -dz/2 to +dz/2
        about the center of the bin).
    low : float
        The lower bound of the integration range.
    high : float
        The upper bound of the integration range.
    dz : float
        The bin width.
    z0 : float
        The center of the first bin.
    """
    z, dndz = nz_binned_to_interp(nz, dz, z0)
    return lin_interp_integral(dndz, z, low, high)


sompz_integral = jax.jit(sompz_integral_nojit, static_argnames=("dz", "z0"))


def lin_interp_integral_nojit(y, x, low, high):
    """Integrate a linearly interpolated set of values
    in a range (low, high).

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
    # ensure integration bounds are ordered
    low, high, sign = jax.lax.cond(
        low < high,
        lambda low, high: (low, high, 1.0),
        lambda low, high: (high, low, -1.0),
        jnp.array(low).astype(jnp.float64),
        jnp.array(high).astype(jnp.float64),
    )

    low = jnp.minimum(x[-1], jnp.maximum(low, x[0]))
    high = jnp.minimum(x[-1], jnp.maximum(high, x[0]))
    low_ind = jnp.digitize(low, x)
    # for the lower index we do not use right=True, but
    # we still want to clip to a valid index of x
    low_ind = jnp.minimum(low_ind, x.shape[0] - 1)
    high_ind = jnp.digitize(high, x, right=True)
    dx = x[1:] - x[:-1]
    m = (y[1:] - y[:-1]) / dx

    # high point not in same bin as low point
    not_in_single_bin = high_ind > low_ind

    ileft = jax.lax.select(
        not_in_single_bin,
        # if not in single bin, this is the fractional bit on the left
        ((y[low_ind - 1] + m[low_ind - 1] * (low - x[low_ind - 1])) + y[low_ind])
        / 2.0
        * (x[low_ind] - low),
        # otherwise this is the whole value
        (
            (y[low_ind - 1] + m[low_ind - 1] * (low - x[low_ind - 1]))
            + (y[low_ind - 1] + m[low_ind - 1] * (high - x[low_ind - 1]))
        )
        / 2.0
        * (high - low),
    )

    # fractional bit on the right
    iright = jax.lax.select(
        not_in_single_bin,
        # if not in single bin, this is the fractional bit on the right
        (
            y[high_ind - 1]
            + (y[high_ind - 1] + m[high_ind - 1] * (high - x[high_ind - 1]))
        )
        / 2.0
        * (high - x[high_ind - 1]),
        # optherwise return 0
        0.0,
    )

    # central bits, if any
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

    return sign * (ileft + icen + iright)


lin_interp_integral = jax.jit(lin_interp_integral_nojit)


def plot_results(*, model_module, model_data, samples=None, map_params=None):
    mn_pars = tuple(tuple(mnp.tolist()) for mnp in model_data["mn_pars"])
    z = model_data["z"]
    nzs = model_data["nz"]
    mn = model_data["mn"]
    cov = model_data["cov"]
    zbins = model_data["zbins"]

    # fmt: off
    array = [
        [1, 3,],
        [1, 3,],
        [1, 3,],
        [1, 3,],
        [2, 4,],
        [5, 7,],
        [5, 7,],
        [5, 7,],
        [5, 7,],
        [6, 8,],
    ]
    # fmt: on

    fig, axs = uplt.subplots(
        array,
        figsize=(8, 6),
        sharex=4,
        sharey=0,
        wspace=None,
        hspace=[0] * 4 + [None] + [0] * 4,
    )

    for bi in range(4):
        # first extract the stats from fit
        if samples is not None:
            ngammas = []
            for i in range(1000):
                _params = {}
                for k, v in samples.items():
                    _params[k] = v[i]
                ngamma = model_module.model_mean_smooth_tomobin(
                    **model_data, tbind=bi, params=_params
                )
                ngammas.append(ngamma)

            ngammas = np.array(ngammas)
            ngamma_mn = np.mean(ngammas, axis=0)
        elif map_params is not None:
            ngammas = np.array(
                [
                    model_module.model_mean_smooth_tomobin(
                        **model_data, tbind=bi, params=map_params
                    )
                ]
            )
            ngamma_mn = ngammas[0]
        else:
            raise ValueError("Either samples or map_params must be provided.")

        ngamma_ints = []
        for ngamma in ngammas:
            ngamma_int = []
            for j in range(10):
                nind = mn_pars.index((j, bi))
                bin_zmin, bin_zmax = zbins[j + 1]
                bin_dz = bin_zmax - bin_zmin
                ngamma_int.append(
                    sompz_integral(ngamma, bin_zmin, bin_zmax) / bin_dz
                )
            ngamma_ints.append(ngamma_int)
        ngamma_ints = np.array(ngamma_ints)

        # get the axes
        bihist = bi * 2
        axhist = axs[bihist]
        bidiff = bi * 2 + 1
        axdiff = axs[bidiff]

        # plot the stuff
        axhist.axhline(0.0, color="black", linestyle="dotted")
        axhist.grid(False)
        axhist.set_yscale("symlog", linthresh=0.2)
        axhist.format(
            xlim=(0, 4.1),
            ylim=(-0.02, 10.0),
            title=f"bin {bi}",
            titleloc="ul",
            xlabel="redshift",
            ylabel=r"redshift density" if bi % 2 == 0 else None,
            yticklabels=[] if bi % 2 == 1 else None,
        )

        axdiff.grid(False)
        axdiff.axhline(0.0, color="black", linestyle="dotted")
        axdiff.format(
            ylim=(-3, 3),
            ylabel=r"(model - data)/error" if bi % 2 == 0 else None,
            yticklabels=[] if bi % 2 == 1 else None,
        )

        axhist.plot(
            z,
            nzs[bi] / DZ,
            drawstyle="steps-mid",
            label=r"$n_{\rm phot}(z)$",
            color="purple",
            linestyle="dashed",
        )
        axhist.plot(
            z, ngamma_mn / DZ, drawstyle="steps-mid", color="black", label=r"$n_\gamma(z)$"
        )
        for i in range(10):
            nind = mn_pars.index((i, bi))
            bin_zmin, bin_zmax = zbins[i + 1]
            bin_dz = bin_zmax - bin_zmin

            nmcal_val = sompz_integral(nzs[bi], bin_zmin, bin_zmax) / bin_dz
            axhist.hlines(
                nmcal_val,
                bin_zmin,
                bin_zmax,
                color="purple",
                linestyle="dashed",
            )

            nga_val = mn[nind] / bin_dz
            nga_err = np.sqrt(cov[nind, nind]) / bin_dz
            axhist.fill_between(
                [bin_zmin, bin_zmax],
                np.ones(2) * nga_val - nga_err,
                np.ones(2) * nga_val + nga_err,
                color="blue",
                alpha=0.5,
            )
            axhist.hlines(
                nga_val,
                bin_zmin,
                bin_zmax,
                color="blue",
                label=r"$N_{\gamma}^{\alpha}$" if i == 0 else None,
            )

            ng_val = np.mean(ngamma_ints, axis=0)[i]
            axhist.hlines(ng_val, bin_zmin, bin_zmax, color="black")
            if ngamma_ints.shape[0] > 1:
                ng_err = np.std(ngamma_ints, axis=0)[i]
                axhist.fill_between(
                    [bin_zmin, bin_zmax],
                    np.ones(2) * ng_val - ng_err,
                    np.ones(2) * ng_val + ng_err,
                    color="black",
                    alpha=0.5,
                )
            axhist.legend(loc="ur", frameon=False, ncols=1)

            axdiff.fill_between(
                [bin_zmin, bin_zmax],
                (np.ones(2) * nga_val - ng_val - nga_err) / nga_err,
                (np.ones(2) * nga_val - ng_val + nga_err) / nga_err,
                color="blue",
                alpha=0.5,
            )
            # axdiff.hlines(
            #     (nga_val - ng_val) / nga_err,
            #     bin_zmin,
            #     bin_zmax,
            #     color="blue",
            # )
            # if ngamma_ints.shape[0] > 1:
            #     ng_err = np.std(ngamma_ints, axis=0)[i]
            #     axdiff.fill_between(
            #         [bin_zmin, bin_zmax],
            #         (np.ones(2) * nga_val - ng_val - ng_err) / ng_err,
            #         (np.ones(2) * nga_val - ng_val + ng_err) / ng_err,
            #         color="black",
            #         alpha=0.5,
            #     )

    return fig


def measure_m_dz(*, model_module, model_data, samples=None, return_dict=False):
    nzs = model_data["nz"]
    n_samples = 1000
    data = np.zeros((8, n_samples))
    for bi in range(4):
        z_nz = compute_nz_binned_mean(nzs[bi])
        assert np.allclose(sompz_integral(nzs[bi], 0.0, 6.0), 1.0)
        for i in range(n_samples):
            _params = {}
            for k, v in samples.items():
                _params[k] = v[i]
            ngamma = model_module.model_mean_smooth_tomobin(
                **model_data, tbind=bi, params=_params
            )
            m = sompz_integral(ngamma, 0.0, 6.0) - 1.0
            z_ngamma = compute_nz_binned_mean(ngamma)

            dz = z_ngamma - z_nz
            data[bi * 2, i] = m
            data[bi * 2 + 1, i] = dz

    if return_dict:
        data = dict(
            m_b0=data[0],
            dz_b0=data[1],
            m_b1=data[2],
            dz_b1=data[3],
            m_b2=data[4],
            dz_b2=data[5],
            m_b3=data[6],
            dz_b3=data[7],
        )
    else:
        data = data.T

    return data
