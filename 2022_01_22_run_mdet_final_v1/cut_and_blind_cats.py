import os
import glob
import fitsio
import io
import sys
import contextlib
import numpy as np
import joblib
from ngmix.shape import (
    e1e2_to_g1g2, g1g2_to_e1e2, g1g2_to_eta1eta2, eta1eta2_to_g1g2
)

from des_y6utils.shear_masking import generate_shear_masking_factor


def _msk_shear(fname, passphrase):
    fac = generate_shear_masking_factor(passphrase)
    failed = False

    try:
        d = fitsio.read(fname)
    except Exception:
        os.system("rm -f " + fname)
        failed = True

    if not failed:
        buff = io.StringIO()
        with contextlib.redirect_stderr(sys.stdout):
            with contextlib.redirect_stdout(buff):
                try:
                    d = fitsio.read(fname)
                    msk = d["flags"] == 0
                    d = d[msk]

                    msk = d["mdet_step"] == "noshear"
                    e1o, e2o = d["mdet_g_1"][msk].copy(), d["mdet_g_2"][msk].copy()
                    g1, g2 = e1e2_to_g1g2(e1o, e2o)
                    eta1, eta2 = g1g2_to_eta1eta2(g1, g2)
                    eta1 *= fac
                    eta2 *= fac
                    g1, g2 = eta1eta2_to_g1g2(eta1, eta2)
                    e1, e2 = g1g2_to_e1e2(g1, g2)
                    d["mdet_g_1"][msk] = e1
                    d["mdet_g_2"][msk] = e2

                    assert not np.array_equal(d["mdet_g_1"][msk], e1o)
                    assert not np.array_equal(d["mdet_g_2"][msk], e2o)

                    out = os.path.join("data_final", os.path.basename(fname))
                    if out.endswith(".fz"):
                        out = out[:-3]
                    fitsio.write(out, d, clobber=True)

                    hs = fname[:-len(".fits.fz")] + "-healsparse-mask.hs"
                    try:
                        os.system("mv %s %s" % (
                            hs,
                            os.path.join("./data_final", os.path.basename(hs))
                        ))
                    except Exception:
                        pass
                except Exception:
                    failed = True
                    pass

    if failed:
        print("tile %s failed!" % fname, flush=True)


os.makedirs("data_final", exist_ok=True)

with open(os.path.expanduser("~/.test_des_blinding_v1"), "r") as fp:
    passphrase = fp.read().strip()

fnames = glob.glob("mdet_data/*.fit*", recursive=True)
jobs = [
    joblib.delayed(_msk_shear)(fname, passphrase)
    for fname in fnames
    # if not os.path.exists("./data_final/" + os.path.basename(fname)[:-3])
]

print("found %d tiles to process" % len(jobs), flush=True)
with joblib.Parallel(n_jobs=1, verbose=100) as exec:
    exec(jobs[0:10])
