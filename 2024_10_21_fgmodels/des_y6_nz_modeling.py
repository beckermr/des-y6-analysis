import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import proplot as pplt  # noqa: E402

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


def gmodel_template_cosmos(z=None):
    if z is None:
        return GMODEL_COSMOS
    else:
        return jnp.interp(z, ZVALS, GMODEL_COSMOS, left=0.0, right=0.0)


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

    fig, axs = pplt.subplots(
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
                    sompz_integral(ngamma, z, bin_zmin, bin_zmax) / bin_dz
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
            nzs[bi],
            drawstyle="steps-mid",
            label=r"$n_{\rm phot}(z)$",
            color="purple",
            linestyle="dashed",
        )
        axhist.plot(
            z, ngamma_mn, drawstyle="steps-mid", color="black", label=r"$n_\gamma(z)$"
        )
        for i in range(10):
            nind = mn_pars.index((i, bi))
            bin_zmin, bin_zmax = zbins[i + 1]
            bin_dz = bin_zmax - bin_zmin

            nmcal_val = sompz_integral(nzs[bi], z, bin_zmin, bin_zmax) / bin_dz
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
