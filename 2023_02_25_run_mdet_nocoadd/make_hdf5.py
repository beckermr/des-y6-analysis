import glob
from esutil.pbar import PBar
import numpy as np
import h5py
import joblib
import gc
import os

import fastparquet

COLS = (
    'tilename',
    'uid',
    'patch_num',
    'mdet_step',
    'ra',
    'dec',
    'gauss_g_1',
    'gauss_g_2',
    'pgauss_band_flux_g',
    'pgauss_band_flux_r',
    'pgauss_band_flux_i',
    'pgauss_band_flux_z',
    'pgauss_band_flux_err_g',
    'pgauss_band_flux_err_r',
    'pgauss_band_flux_err_i',
    'pgauss_band_flux_err_z',
    'gauss_T_err',
    'gauss_T_ratio',
    'gauss_psf_T',
    'gauss_s2n',
    'slice_id',
    'x',
    'y',
    'mfrac',
    'mfrac_img',
    'nepoch_g',
    'nepoch_r',
    'nepoch_i',
    'nepoch_z',
    'psfrec_g_1',
    'psfrec_g_2',
    'psfrec_T',
    'gauss_g_cov_1_1',
    'gauss_g_cov_1_2',
    'gauss_g_cov_2_2',
    'pgauss_T_err',
    'pgauss_T',
    'pgauss_psf_T',
)


def _get_col_one(fname, cols, mdet_step):
    pf = fastparquet.ParquetFile(fname)
    return pf.to_pandas(
        columns=cols,
        filters=[[("mdet_step", "==", mdet_step)]],
        row_filter=True,
    ).to_records(
        index=False,
        column_dtypes={"tilename": "<S12", "mdet_step": "<S7"}
    )[cols]


def _get_col(fnames, cols, mdet_step, n_jobs=8):
    print(" ", flush=True)
    jobs = [
        joblib.delayed(_get_col_one)(fname, cols, mdet_step)
        for fname in fnames
    ]
    with joblib.Parallel(n_jobs=n_jobs, verbose=10) as par:
        return np.concatenate(par(jobs), axis=0)


ofile_hdf5 = "metadetect_desdmv5a_cutsv5.h5"
pf_fnames = glob.glob("cutsv5/patch-*.parq")

os.system("rm -f %s" % ofile_hdf5)

pf = fastparquet.ParquetFile(pf_fnames[0])
d = pf.to_pandas(
    filters=[[("mdet_step", "==", "noshear")]],
    row_filter=True,
).to_records(
    index=False,
    column_dtypes={"tilename": "<S12", "mdet_step": "<S7"}
)

chunk_size = 2
nchunks = len(COLS) // chunk_size
if nchunks * chunk_size < len(COLS):
    nchunks += 1

with h5py.File(ofile_hdf5, "w") as fp:
    mdet_grp = fp.create_group("mdet")
    for mdet_step in PBar(
        ["noshear", "1p", "1m", "2p", "2m"],
        desc="mdet step",
    ):
        grp = mdet_grp.create_group(mdet_step)
        for col in d.dtype.names:
            dt = d[col].dtype
            grp.create_dataset(
                col,
                dtype=dt,
                shape=(150_000_000,),
                maxshape=(None,),
            )

for mdet_step in PBar(["noshear", "1p", "1m", "2p", "2m"], desc="mdet step"):
    loc = 0
    for chunk in PBar(range(nchunks), desc="%s columns" % mdet_step):
        max_loc = min(loc + chunk_size, len(COLS))
        cols = list(COLS[loc:max_loc])
        d = _get_col(pf_fnames, cols, mdet_step)
        for col in cols:
            with h5py.File(ofile_hdf5, "a") as fp:
                fp["mdet/" + mdet_step][col].resize(len(d[col]), axis=0)
                fp["mdet/" + mdet_step][col][:] = d[col]
        del fp
        del d
        gc.collect()
        loc += chunk_size
