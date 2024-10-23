import numpy as np

from des_y6_nz_modeling import gmodel_template_cosmos, fmodel_mstudt4, ZVALS


def test_gmodel_template_cosmos():
    assert gmodel_template_cosmos[0] == 0.0


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
