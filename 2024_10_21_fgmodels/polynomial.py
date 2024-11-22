import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from des_y6_nz_modeling import (  # noqa: E402
    sompz_integral,
)

POLY_ORDER = 3


def model_mean_smooth(*, mu, sigma, z, nz, mn_pars, zbins, params, mn=None, cov=None):
    ngammas = []
    for i in range(4):
        c = []
        for j in range(POLY_ORDER + 1):
            c.append(params.get(f"c{j}_b{i}", 0.0))

        zs = (z - mu[i]) / sigma[i]
        ngamma = nz[i]
        for j in range(POLY_ORDER + 1):
            ngamma = ngamma + c[j] * jnp.power(zs, j)

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
        val = sompz_integral(ngammas[bi], z, zlow, zhigh)
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

    cwidth = numpyro.sample("c_sigma_coeff", dist.LogUniform(0.001, 10.0))
    params = {}
    for i in range(4):
        for j in range(POLY_ORDER + 1):
            params[f"c{j}_b{i}"] = numpyro.sample(
                f"c{j}_b{i}", dist.Normal(0.0, cwidth)
            )

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
    params = []
    for i in range(4):
        _nrm = sompz_integral(nzs[i], z, 0, 6)
        _mn = sompz_integral(z * nzs[i], z, 0, 6) / _nrm
        _sd = jnp.sqrt(sompz_integral(nzs[i] * jnp.square(z - _mn), z, 0, 6) / _nrm)
        params.append(
            (
                _mn,
                _sd,
            )
        )
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
