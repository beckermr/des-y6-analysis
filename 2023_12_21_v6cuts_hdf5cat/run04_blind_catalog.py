#!/usr/bin/env python

import os
import io
import sys
from des_y6utils.shear_masking import generate_shear_masking_factor
from ngmix.shape import g1g2_to_eta1eta2, eta1eta2_to_g1g2
import contextlib
import h5py
import numpy as np

COLS_TO_KEEP = ["pgauss", "gauss"]

ofile_hdf5 = "metadetect_cutsv6_all.h5"
bofile_hdf5 = ofile_hdf5.rsplit(".", maxsplit=1)[0] + "_blinded.h5"
print(f"blinding {ofile_hdf5} to {bofile_hdf5}...", flush=True)

with open(os.path.expanduser("~/.test_des_blinding_v7"), "r") as fp:
        passphrase = fp.read().strip()

fac = generate_shear_masking_factor(passphrase)

print("making a copy...", flush=True)
os.system(f"rm -f {bofile_hdf5}")
os.system(f"cp {ofile_hdf5} {bofile_hdf5}")

print("blinding the data...", flush=True)
buff = io.StringIO()
with contextlib.redirect_stderr(sys.stdout):
    with contextlib.redirect_stdout(buff):
        try:
            with h5py.File(bofile_hdf5, "a") as fp:
                for pre in COLS_TO_KEEP:
                    e1o, e2o = (
                        fp["mdet"]["noshear"][pre + "_g_1"][:].copy(),
                        fp["mdet"]["noshear"][pre + "_g_2"][:].copy(),
                    )
                    if pre not in ["gauss"]:
                        e1 = e1o * fac
                        e2 = e2o * fac
                    else:
                        # use eta due to bounds
                        eta1o, eta2o = g1g2_to_eta1eta2(e1o, e2o)
                        eta1 = eta1o * fac
                        eta2 = eta2o * fac
                        e1, e2 = eta1eta2_to_g1g2(eta1, eta2)

                    fp["mdet"]["noshear"][pre + "_g_1"][:] = e1
                    fp["mdet"]["noshear"][pre + "_g_2"][:] = e2

                    fp.flush()

                    assert not np.array_equal(fp["mdet"]["noshear"][pre + "_g_1"][:], e1o)
                    assert not np.array_equal(fp["mdet"]["noshear"][pre + "_g_2"][:], e2o)

        except Exception:
            failed = True
            print("blinding error", flush=True)

os.system(f"chmod go+r {bofile_hdf5}")
