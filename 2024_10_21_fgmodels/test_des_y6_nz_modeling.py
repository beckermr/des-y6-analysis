import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline

import pytest

from des_y6_nz_modeling import (
    gmodel_template_cosmos,
    fmodel_mstudt4,
    ZVALS,
    sompz_integral,
    sompz_integral_nojit,
)


def test_gmodel_template_cosmos():
    assert gmodel_template_cosmos()[0] == 0.0


def test_fmodel_mstudt4():
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    a4 = 0.0
    mu = 0.5
    sigma = 0.3
    assert np.all(fmodel_mstudt4(ZVALS, a0, a1, a2, a3, a4, mu, sigma) == 0.0)
    assert np.all(fmodel_mstudt4(ZVALS, 0.3, a1, a2, a3, a4, mu, sigma) == 0.3)


@pytest.mark.parametrize("func", [sompz_integral, sompz_integral_nojit])
def test_sompz_integral(func):
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 0.8, 100))
    y = np.sin(x)

    spl = InterpolatedUnivariateSpline(x, y, k=1, ext=1)

    # first test whole range
    assert_allclose(
        np.trapz(y, x),
        func(y, x, -10, 10),
        rtol=0,
        atol=1e-16,
    )
    assert_allclose(
        np.trapz(y, x),
        func(y, x, 0, 0.8),
        rtol=0,
        atol=1e-16,
    )
    assert_allclose(
        spl.integral(-10, 10),
        func(y, x, -10, 10),
        rtol=0,
        atol=1e-16,
    )
    assert_allclose(
        spl.integral(0, 0.8),
        func(y, x, 0, 0.8),
        rtol=0,
        atol=1e-16,
    )

    # now try out side the range on both sides
    for low, high in [(-10, -1), (-3.4, 0), (1, 10), (50, 100)]:
        assert_allclose(
            0.0,
            func(y, x, low, high),
            rtol=0,
            atol=1e-16,
        )

    # now try random ranges in the middle
    for low, high in [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7)]:
        assert_allclose(
            spl.integral(low, high),
            func(y, x, low, high),
            rtol=0,
            atol=1e-4,
        )

    # now try inside bins
    for ind in [0, 4, 98]:
        dx = x[ind + 1] - x[ind]
        low = x[ind]
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            high = x[ind] + dx * fac
            assert_allclose(
                spl.integral(low, high),
                func(y, x, low, high),
                rtol=0,
                atol=1e-4,
            )

        low = x[ind] + 0.05 * dx
        for fac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            high = x[ind] + dx * fac
            assert_allclose(
                spl.integral(low, high),
                func(y, x, low, high),
                rtol=0,
                atol=1e-4,
            )
