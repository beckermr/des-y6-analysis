"""DO NOT USE"""
assert False, "Do not use this code."
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from des_y6_nz_modeling import (  # noqa: E402
    sompz_integral,
    GMODEL_COSMOS_NZ,
)

NPTS = 10


def model_parts_smooth(
    *, params, pts, z, nz=None, mn_pars=None, zbins=None, mn=None, cov=None
):
    gtemp = GMODEL_COSMOS_NZ[:z.shape[0]]
    gtemp = gtemp / gtemp.sum()
    model_parts = {}
    for i in range(4):
        model_parts[i] = {}
        ypts = [0.0]
        for j in range(NPTS):
            ypts.append(params[f"a{j}_b{i}"])
        ypts.append(0.0)
        y_pts = jnp.asarray(ypts)

        x_pts = pts[i]
        akint = jnp.interp(z, x_pts, y_pts, left=0.0, right=0.0)
        model_parts[i]["F"] = akint

        g = params.get(f"g_b{i}", 0.0)
        model_parts[i]["G"] = g * gtemp

    return model_parts


def model_mean_smooth(*, pts, z, nz, mn_pars, zbins, params, mn=None, cov=None):
    model_parts = model_parts_smooth(
        pts=pts,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )
    ngammas = []
    for i in range(4):
        ngamma = (1.0 + model_parts[i]["F"]) * nz[i] + model_parts[i]["G"]
        ngammas.append(ngamma)

    return jnp.stack(ngammas)


def model_mean(*, pts, z, nz, mn_pars, zbins, params, mn=None, cov=None):
    ngammas = model_mean_smooth(
        pts=pts,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )

    def _scan_func(mn_pars, ind):
        si, bi = mn_pars[ind]
        zlow, zhigh = zbins[si + 1]
        val = sompz_integral(ngammas[bi], zlow, zhigh)
        return mn_pars, val

    inds = jnp.arange(len(mn_pars))
    _, model = jax.lax.scan(_scan_func, mn_pars, inds)
    return model


def model_mean_smooth_tomobin(
    *, pts, z, nz, mn_pars, zbins, params, tbind, mn=None, cov=None
):
    model_mn = model_mean_smooth(
        pts=pts, z=z, nz=nz, mn_pars=mn_pars, zbins=zbins, params=params
    )
    return np.asarray(model_mn)[tbind]


def model(pts=None, z=None, nz=None, mn=None, cov=None, mn_pars=None, zbins=None):
    assert pts is not None
    assert nz is not None
    assert mn is not None
    assert cov is not None
    assert mn_pars is not None
    assert zbins is not None
    assert z is not None

    params = {}
    for i in range(4):
        # params[f"g_b{i}"] = numpyro.sample(f"g_b{i}", dist.Normal(0.0, 10))
        for j in range(NPTS):
            params[f"a{j}_b{i}"] = numpyro.sample(f"a{j}_b{i}", dist.Uniform(-10, 10))

    model_mn = model_mean(
        pts=pts,
        z=z,
        nz=nz,
        mn_pars=mn_pars,
        zbins=zbins,
        params=params,
    )
    numpyro.sample(
        "model", dist.MultivariateNormal(loc=model_mn, covariance_matrix=cov), obs=mn
    )


def make_model_data(*, z, nzs, mn, cov, mn_pars, zbins):
    """Create the dict of model data.

    Parameters
    ----------
    z : array
        The redshift values.
    nzs : dict mapping bin index to n(z).
        The input n(z) data.
    mn : array
        The measured N_gamma_alpha values.
    cov : array
        The covariance matrix for the measured N_gamma_alpha values.
    mn_pars : array
        The mapping of N_gamma_alpha values to (shear bin, tomographic bin)
        indices.
    zbins : array
        The shear bin edges.

    Returns
    -------
    data : dict
        The model data. Pass to the functions using `**data`.
    """
    pts = []
    for i in range(4):
        msk = z < 2.7
        frac = np.cumsum(nzs[i][msk])
        dfrac = 1.0 / (NPTS - 1)
        pvals = np.arange(NPTS-1) * dfrac + dfrac / 2.0
        zmid = np.interp(pvals, frac, z[msk])
        pts.append(
            np.concatenate(
                [
                    np.array([0.0]),
                    zmid,
                    [(6+2.7)/2.0],
                    np.array([6.01]),
                ]
            )
        )

    return dict(
        pts=np.array(pts),
        z=z,
        nz=nzs,
        mn=mn,
        cov=cov,
        mn_pars=jnp.asarray(mn_pars, dtype=np.int32),
        zbins=jnp.asarray(zbins),
    )
