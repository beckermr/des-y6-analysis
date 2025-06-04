import jax

jax.config.update("jax_enable_x64", True)

import functools  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from des_y6_nz_modeling import (  # noqa: E402
    sompz_integral,
)


def _trunc(z, scale=0.01, trunc_scale=1.0):
    return jax.nn.sigmoid((z - scale / 2) / (scale / 100)) * jnp.exp(-z / trunc_scale)


@jax.jit
def _akima_interp_coeffs_jax(x, y):
    dx = x[1:] - x[:-1]
    mi = (y[1:] - y[:-1]) / dx

    # these values are imposed for points
    # at the ends
    s0 = mi[0:1]
    s1 = (mi[0:1] + mi[1:2]) / 2.0
    snm2 = (mi[-3:-2] + mi[-2:-1]) / 2.0
    snm1 = mi[-2:-1]

    wim1 = jnp.abs(mi[3:] - mi[2:-1])
    wi = jnp.abs(mi[1:-2] - mi[0:-3])
    denom = wim1 + wi
    numer = wim1 * mi[1:-2] + wi * mi[2:-1]

    smid = jnp.where(
        jnp.abs(denom) >= 1e-12,
        numer / denom,
        (mi[1:-2] + mi[2:-1]) / 2.0,
    )
    s = jnp.concatenate([s0, s1, smid, snm2, snm1])

    # these coeffs are for
    # P(x) = a + b * (x-xi) + c * (x-xi)**2 + d * (x-xi)**3
    # for a point x that falls in [xi, xip1]
    a = y[:-1]
    b = s[:-1]
    c = (3 * mi - 2 * s[:-1] - s[1:]) / dx
    d = (s[:-1] + s[1:] - 2 * mi) / dx / dx

    return (a, b, c, d)


@functools.partial(jax.jit, static_argnames=("fixed_spacing",))
def _akima_interp(x, xp, yp, coeffs, fixed_spacing=False):
    """Conmpute the values of an Akima cubic spline at a set of points given the
    interpolation coefficients.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the points where the interpolation is computed.
    xp : array-like
        The x-coordinates of the data points. These must be sorted into increasing order
        and cannot contain any duplicates.
    yp : array-like
        The y-coordinates of the data points. Not used currently.
    coeffs : tuple
        The interpolation coefficients returned by `akima_interp_coeffs`.
    fixed_spacing : bool, optional
        Whether the data points are evenly spaced. Default is False. If True, the
        code uses a faster technique to compute the index of the data points x into
        the array xp such that xp[i] <= x < xp[i+1].

    Returns
    -------
    array-like
        The values of the Akima cubic spline at the points x.
    """
    xp = jnp.asarray(xp)
    # yp = jnp.array(yp)  # unused
    if fixed_spacing:
        dxp = xp[1] - xp[0]
        i = jnp.floor((x - xp[0]) / dxp).astype(jnp.int32)
        i = jnp.clip(i, 0, len(xp) - 2)
    else:
        i = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1) - 1

    # these coeffs are for
    # P(x) = a + b * (x-xi) + c * (x-xi)**2 + d * (x-xi)**3
    # for a point x that falls in [xi, xip1]
    a, b, c, d = coeffs
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    c = jnp.asarray(c)
    d = jnp.asarray(d)

    dx = x - xp[i]
    dx2 = dx * dx
    dx3 = dx2 * dx
    xval = a[i] + b[i] * dx + c[i] * dx2 + d[i] * dx3

    xval = jnp.where(x < xp[0], 0, xval)
    xval = jnp.where(x > xp[-1], 0, xval)
    return xval


def model_mean_smooth(*, pts, z, nz, mn_pars, zbins, params, mn=None, cov=None):
    ngammas = []
    for i in range(4):
        a0 = params.get(f"a0_b{i}", 0.0)
        a1 = params.get(f"a1_b{i}", 0.0)
        a2 = params.get(f"a2_b{i}", 0.0)
        a3 = params.get(f"a3_b{i}", 0.0)
        a4 = params.get(f"a4_b{i}", 0.0)
        a5 = params.get(f"a5_b{i}", 0.0)
        a6 = params.get(f"a6_b{i}", 0.0)
        a7 = params.get(f"a7_b{i}", 0.0)
        a8 = params.get(f"a8_b{i}", 0.0)
        a9 = params.get(f"a9_b{i}", 0.0)
        y_pts = jnp.asarray([0.0, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, 0.0])
        x_pts = pts[i]
        coeffs = _akima_interp_coeffs_jax(x_pts, y_pts)
        akint = _akima_interp(z, x_pts, y_pts, coeffs, fixed_spacing=False)
        # akint = jnp.interp(z, x_pts, y_pts)

        ngamma = nz[i] + akint
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
        val = sompz_integral(ngammas[bi], z, zlow, zhigh)
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

    areg = numpyro.sample("a_reg", dist.LogUniform(0.001, 10.0))
    prior = dist.Normal(0.0, areg)
    params = {}
    for i in range(4):
        params[f"a0_b{i}"] = numpyro.sample(f"a0_b{i}", prior)
        params[f"a1_b{i}"] = numpyro.sample(f"a1_b{i}", prior)
        params[f"a2_b{i}"] = numpyro.sample(f"a2_b{i}", prior)
        params[f"a3_b{i}"] = numpyro.sample(f"a3_b{i}", prior)
        params[f"a4_b{i}"] = numpyro.sample(f"a4_b{i}", prior)
        params[f"a5_b{i}"] = numpyro.sample(f"a5_b{i}", prior)
        params[f"a6_b{i}"] = numpyro.sample(f"a6_b{i}", prior)
        params[f"a7_b{i}"] = numpyro.sample(f"a7_b{i}", prior)
        params[f"a8_b{i}"] = numpyro.sample(f"a8_b{i}", prior)
        params[f"a9_b{i}"] = numpyro.sample(f"a9_b{i}", prior)

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
        # nrm = sompz_integral(nzs[i], z, 0, 6)
        # zvals = np.linspace(0, 6, 1000)
        # pvals = np.array(
        #     [sompz_integral(nzs[i], z, 0, zbin) / nrm for zbin in zvals]
        # )
        # pts.append(
        #     np.concatenate([
        #         np.array([0.0]),
        #         np.interp(
        #             np.array([0.05, 0.2, 0.35, 0.5, 0.65, 0.80, 0.95]),
        #             pvals,
        #             zvals,
        #         ),
        #         np.array([6.0]),
        #     ])
        # )
        pts.append(
            np.concatenate(
                [
                    np.array([0.0]),
                    [(zb[0] + zb[1]) / 2 for zb in zbins[1:]],
                    np.array([6.0]),
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
