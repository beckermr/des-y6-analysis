import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.optimize  # noqa: E402


DZ = 0.01
ZVALS = np.linspace(0, 3, 301)[:-1]
ZLOW = ZVALS - DZ / 2
ZHIGH = ZVALS + DZ / 2
# fmt: off
GMODEL_COSMOS = np.array([
    0.        , 0.07357269, 0.04144495, 0.04716873, 0.13497737,
    0.15113352, 0.08032746, 0.6763368 , 3.08229478, 1.40066704,
    1.08022809, 1.13240298, 1.56979355, 0.8056257 , 2.13085256,
    0.73490781, 0.82937835, 2.79326472, 0.78989048, 1.02830903,
    4.93469563, 1.49296567, 2.12113   , 4.86320227, 0.40055684,
    1.27967162, 1.77483015, 1.67678962, 0.79168667, 0.68660956,
    1.00059222, 1.32911608, 1.3094135 , 1.67509146, 2.19565736,
    1.68272676, 1.8278239 , 2.01967052, 2.31094884, 0.8912197 ,
    0.62753701, 0.80097969, 0.7401859 , 0.93904417, 0.77945397,
    0.46577696, 0.60308055, 0.59060266, 0.69294554, 0.648369  ,
    0.76781554, 0.66611127, 0.71854175, 0.78331368, 0.58017585,
    0.48295985, 0.47488454, 0.44103787, 0.30164353, 0.51568479,
    0.70971155, 0.63539341, 0.47718154, 0.48440872, 0.36744416,
    0.44961394, 1.1084875 , 0.9931591 , 0.61668847, 0.62128452,
    0.82409833, 0.55884677, 0.66944163, 0.61781953, 0.55752277,
    0.49599724, 0.40919192, 0.26095548, 0.31379402, 0.43348146,
    0.39703839, 0.42272548, 0.53130346, 0.6043246 , 0.54234357,
    0.5109659 , 0.54074776, 0.45621326, 0.3566659 , 0.39910516,
    0.40582442, 0.43439884, 0.33904144, 0.5930522 , 0.42726077,
    0.45453682, 0.45901591, 0.39739646, 0.40891439, 0.38041658,
    0.31866661, 0.2581774 , 0.18096474, 0.11972662, 0.14738275,
    0.14501299, 0.1443507 , 0.16714173, 0.19189187, 0.19867568,
    0.18423301, 0.20504553, 0.20489188, 0.19252087, 0.21966395,
    0.24615169, 0.15185047, 0.16094615, 0.17887102, 0.24850844,
    0.13290574, 0.15240944, 0.18563243, 0.22594396, 0.19736746,
    0.16265058, 0.1604313 , 0.15766817, 0.12782448, 0.10439763,
    0.08760031, 0.09056818, 0.08356258, 0.09046451, 0.08928613,
    0.0998872 , 0.08718349, 0.11580666, 0.0936604 , 0.0909831 ,
    0.09025291, 0.10593989, 0.11138645, 0.10074247, 0.09984696,
    0.10683368, 0.09899665, 0.10241146, 0.0978462 , 0.07949943,
    0.07191443, 0.07506805, 0.06176384, 0.05536867, 0.05778376,
    0.0528917 , 0.05962365, 0.05558557, 0.06200207, 0.06331176,
    0.061042  , 0.05925497, 0.07724345, 0.0807462 , 0.05968625,
    0.05055174, 0.0440703 , 0.04261823, 0.03700904, 0.03403373,
    0.03713492, 0.03527725, 0.04105227, 0.05236657, 0.05855507,
    0.0511013 , 0.04704811, 0.04482291, 0.05300909, 0.04893563,
    0.05147051, 0.05165306, 0.04682791, 0.04670618, 0.03833204,
    0.03975585, 0.03051012, 0.03583409, 0.03075184, 0.03146492,
    0.02986222, 0.03568887, 0.03768597, 0.04047338, 0.04096569,
    0.04502993, 0.04026349, 0.04077783, 0.04510326, 0.04357238,
    0.0403208 , 0.03264111, 0.03231458, 0.02714381, 0.02997765,
    0.02778648, 0.02794257, 0.02824192, 0.02268953, 0.02261284,
    0.02468242, 0.0234018 , 0.02468485, 0.02838441, 0.0255049 ,
    0.02953897, 0.0279913 , 0.02507162, 0.02125422, 0.02296056,
    0.02292003, 0.01963778, 0.02114082, 0.02131119, 0.02117851,
    0.02281275, 0.02079052, 0.0237365 , 0.02223195, 0.02143741,
    0.02349697, 0.02143483, 0.02564389, 0.02086926, 0.02021896,
    0.01935243, 0.02178976, 0.02107996, 0.02042578, 0.01821144,
    0.02038183, 0.01644049, 0.01458929, 0.01497151, 0.01413422,
    0.01484769, 0.01857645, 0.01672486, 0.02160071, 0.02176701,
    0.0205252 , 0.01874631, 0.02040696, 0.02283233, 0.01937174,
    0.0220419 , 0.01692309, 0.01925976, 0.01571336, 0.01443104,
    0.01489348, 0.01530922, 0.0170521 , 0.01532869, 0.01989535,
    0.01743284, 0.01937491, 0.02182629, 0.01952026, 0.01915206,
    0.01913092, 0.01891412, 0.01994567, 0.01727781, 0.01687509,
    0.02132841, 0.0201657 , 0.02040193, 0.01958608, 0.01920378,
    0.0214005 , 0.0207628 , 0.02209657, 0.01759594, 0.01730302,
    0.01525733, 0.0150203 , 0.01583391, 0.01296612, 0.01471473,
    0.01210726, 0.01258789, 0.01713214, 0.01744618, 0.01588308,
    0.01592441, 0.01628888, 0.01357622, 0.0175545 , 0.01558222], dtype=np.float64)
# fmt: on
ZBIN_LOW = np.array([0., 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])
ZBIN_HIGH = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0])


def mstudt_trunc(z):
    return jax.nn.sigmoid((z - DZ/2) / (DZ / 100))

@jax.jit
def mstudt(z, mu, sigma):
    nu = 3
    znrm = (z - mu) / sigma
    vals = jnp.power(1 + znrm * znrm / nu, -(nu + 1)/2)
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
    return vals / jnp.sum(vals)


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
            nzs[i] / np.sum(nzs[i]),
            p0=(1, 0.1) if popt is None else popt,
        )
        params[i] = tuple(v for v in popt)

    return params