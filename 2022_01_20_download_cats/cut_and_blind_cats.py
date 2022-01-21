import os
import glob
import fitsio
import io
import contextlib
import numpy as np
import tqdm
from ngmix.shape import (
    e1e2_to_g1g2, g1g2_to_e1e2, g1g2_to_eta1eta2, eta1eta2_to_g1g2
)

from des_y6utils.shear_masking import generate_shear_masking_factor

os.makedirs("data_final", exist_ok=True)

with open(os.path.expanduser("~/.test_des_blinding_v1"), "r") as fp:
    passphrase = fp.read().strip()

fac = generate_shear_masking_factor(passphrase)

fnames = glob.glob("data/**/*.fit*", recursive=True)

for fname in tqdm.tqdm(fnames):
    buff = io.StringIO()
    with contextlib.redirect_stdout(buff):
        with contextlib.redirect_stderr(buff):
            try:
                d = fitsio.read(fname)
                msk = d["flags"] == 0
                d = d[msk]

                msk = d["shear_step"] == "no_shear"
                e1o, e2o = d["mdet_g_1"][msk], d["mdet_g_2"][msk]
                g1, g2 = e1e2_to_g1g2(e1o, e2o)
                eta1, eta2 = g1g2_to_eta1eta2(g1, g2)
                eta1 *= fac
                eta2 *= fac
                g1, g2 = eta1eta2_to_g1g2(eta1, eta2)
                e1, e2 = g1g2_to_e1e2(g1, g2)
                d["mdet_g_1"][msk] = e1
                d["mdet_g_2"][msk] = e2

                assert not np.allclose(d["mdet_g_1"][msk], e1o)
                assert not np.allclose(d["mdet_g_2"][msk], e2o)
                out = os.path.join("data_final", os.path.basename(fname))
                fitsio.write(out, d, clobber=True)
            except Exception:
                raise RuntimeError("failed!")
