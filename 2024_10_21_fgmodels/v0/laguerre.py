import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from des_y6_nz_modeling import (  # noqa: E402
    gmodel_template_cosmos,
    sompz_integral,
)


def model_mean_smooth(*, z, nz, mn_pars, zbins, params, mn=None, cov=None):
    ngammas = []
    for i in range(4):
        a0 = params.get(f"a0_b{i}", 0.0)
        a1 = params.get(f"a1_b{i}", 0.0)
        a2 = params.get(f"a2_b{i}", 0.0)
        a3 = params.get(f"a3_b{i}", 0.0)
        a4 = params.get(f"a4_b{i}", 0.0)
        a5 = params.get(f"a5_b{i}", 0.0)
        g_z0 = params.get("g_z0", 0.0)
        g_alpha = params.get("g_alpha", 0.0)
        sigma = params.get(f"sigma_b{i}", 0.3)

        _nrm = sompz_integral(nz[i], z, 0, 6)
        _mn = sompz_integral(z * nz[i], z, 0, 6) / _nrm
        zs = z / sigma

        g = g_z0 * jnp.power(1.0 + _mn, g_alpha)

        ngamma = (
            nz[i]
            + (
                a0
                + a1 * zs
                + a2 * (zs**2 - 4 * zs + 2) / 2.0
                - 1.0
                - a3 * (zs**3 - 9 * zs**2 + 18 * zs - 6) / 6.0
                - 1.0
                + a4 * (zs**4 - 16 * zs**3 + 72 * zs**2 - 96 * zs + 24) / 24.0
                - 1.0
                - a5
                * (zs**5 - 25 * zs**4 + 200 * zs**3 - 600 * zs**2 + 600 * zs + 120)
                / 120.0
                - 1.0
            )
            * jnp.exp(-zs)
            / sigma
            + g * gmodel_template_cosmos()
        )
        ngammas.append(ngamma)

    return jnp.stack(ngammas)


def model_mean(*, z, nz, mn_pars, zbins, params, mn=None, cov=None):
    ngammas = model_mean_smooth(
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
    *, z, nz, mn_pars, zbins, params, tbind, mn=None, cov=None
):
    model_mn = model_mean_smooth(
        z=z, nz=nz, mn_pars=mn_pars, zbins=zbins, params=params
    )
    return np.asarray(model_mn)[tbind]


def model(z=None, nz=None, mn=None, cov=None, mn_pars=None, zbins=None):
    assert nz is not None
    assert mn is not None
    assert cov is not None
    assert mn_pars is not None
    assert zbins is not None
    assert z is not None

    awidth = numpyro.sample("a_sigma_coeff", dist.LogUniform(0.001, 10.0))
    gwidth = numpyro.sample("g_sigma_coeff", dist.LogUniform(0.001, 10.0))
    params = {}
    params["g_z0"] = numpyro.sample("g_z0", dist.Normal(0.0, gwidth))
    params["g_alpha"] = numpyro.sample("g_alpha", dist.Uniform(-10, 10))
    for i in range(4):
        params[f"a0_b{i}"] = numpyro.sample(f"a0_b{i}", dist.Normal(0.0, awidth))
        params[f"a1_b{i}"] = numpyro.sample(f"a1_b{i}", dist.Normal(0.0, awidth))
        params[f"a2_b{i}"] = numpyro.sample(f"a2_b{i}", dist.Normal(0.0, awidth))
        params[f"a3_b{i}"] = numpyro.sample(f"a3_b{i}", dist.Normal(0.0, awidth))
        params[f"a4_b{i}"] = numpyro.sample(f"a4_b{i}", dist.Normal(0.0, awidth))
        params[f"a5_b{i}"] = numpyro.sample(f"a5_b{i}", dist.Normal(0.0, awidth))

        params[f"sigma_b{i}"] = numpyro.sample(
            f"sigma_b{i}", dist.LogUniform(0.2, 10.0)
        )

    model_mn = model_mean(
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
    return dict(
        z=z,
        nz=nzs,
        mn=mn,
        cov=cov,
        mn_pars=jnp.asarray(mn_pars, dtype=np.int32),
        zbins=jnp.asarray(zbins),
    )
