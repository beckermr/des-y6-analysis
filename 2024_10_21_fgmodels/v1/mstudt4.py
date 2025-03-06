import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.optimize  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from des_y6_nz_modeling import (  # noqa: E402
    compute_nz_binned_mean,
    sompz_integral,
    sompz_integral_nojit,
    GMODEL_COSMOS_NZ,
)


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
    return vals / sompz_integral_nojit(vals, 0, 6)


def fit_nz_data_for_template_params(z, nzs):
    """Fit the modified Student's t model to the n(z) data.

    Parameters
    ----------
    z : array
        The redshift values.
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
            z,
            nzs[i] / sompz_integral(nzs[i], 0, 6),
            p0=(1, 0.1) if popt is None else popt,
        )
        params[i] = tuple(v for v in popt)

    return params


def model_mean_smooth(*, mu, sigma, z, nz, mn_pars, zbins, params, mn=None, cov=None):
    ngammas = []
    for i in range(4):
        a0 = params.get(f"a0_b{i}", 0.0)
        a1 = params.get(f"a1_b{i}", 0.0)
        a2 = params.get(f"a2_b{i}", 0.0)
        a3 = params.get(f"a3_b{i}", 0.0)
        a4 = params.get(f"a4_b{i}", 0.0)
        g = params.get(f"g_b{i}", 0.0)
        dmu_i = params[f"dmu_b{i}"]
        sigma_i = params[f"sigma_b{i}"]

        _mn = compute_nz_binned_mean(nz[i])

        mu_i = _mn + dmu_i

        fmod = fmodel_mstudt4(z, a0, a1, a2, a3, a4, mu_i, sigma_i)
        gmod = g * GMODEL_COSMOS_NZ[:nz[i].shape[0]]
        ngamma = (1.0 + fmod) * nz[i] + gmod

        ngammas.append(ngamma)

    return jnp.stack(ngammas)


def model_mean(*, mu, sigma, z, nz, mn_pars, zbins, params, mn=None, cov=None):
    ngammas = model_mean_smooth(
        mu=mu,
        sigma=sigma,
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
    *, mu, sigma, z, nz, mn_pars, zbins, params, tbind, mn=None, cov=None
):
    model_mn = model_mean_smooth(
        mu=mu, sigma=sigma, z=z, nz=nz, mn_pars=mn_pars, zbins=zbins, params=params
    )
    return np.asarray(model_mn)[tbind]


def model(
    mu=None, sigma=None, z=None, nz=None, mn=None, cov=None, mn_pars=None, zbins=None
):
    assert mu is not None
    assert sigma is not None
    assert nz is not None
    assert mn is not None
    assert cov is not None
    assert mn_pars is not None
    assert zbins is not None
    assert z is not None

    params = {}
    for i in range(4):
        params[f"dmu_b{i}"] = numpyro.sample(f"dmu_b{i}", dist.Normal(0.0, 0.1))
        lns = 0.5
        params[f"sigma_b{i}"] = numpyro.sample(
            f"sigma_b{i}", dist.LogNormal(np.log(0.8) - lns**2 / 2, lns)
        )
        for j in range(3):
            params[f"a{j}_b{i}"] = numpyro.sample(f"a{j}_b{i}", dist.Normal(0.0, 0.1))

        params[f"g_b{i}"] = numpyro.sample(f"g_b{i}", dist.Normal(0.0, 0.1))

    model_mn = model_mean(
        mu=mu,
        sigma=sigma,
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
    params = fit_nz_data_for_template_params(z, nzs)
    return dict(
        mu=tuple(params[i][0] for i in range(4)),
        sigma=tuple(params[i][1] for i in range(4)),
        z=z,
        nz=nzs,
        mn=mn,
        cov=cov,
        mn_pars=jnp.asarray(mn_pars, dtype=np.int32),
        zbins=jnp.asarray(zbins),
    )
