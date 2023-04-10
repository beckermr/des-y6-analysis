import glob
import os

import fitsio
import joblib
import pandas as pd
import fastparquet
from numpy.lib.recfunctions import repack_fields
from des_y6utils.mdet import make_mdet_cuts


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


def _reformat_one(fname, odir):
    d = fitsio.read(fname)
    msk = make_mdet_cuts(d, "5")
    d = d[msk]
    d = repack_fields(d[[col for col in COLS if col in d.dtype.names]])

    d = pd.DataFrame(d)
    fastparquet.write(
        os.path.join(odir, os.path.basename(fname)[:-len(".fits")] + ".parq"),
        d,
        has_nulls=False,
        write_index=False,
        compression="SNAPPY",
    )


def _reformat(fnames, odir, n_jobs=4):
    os.system("rm -rf %s/*" % odir)
    os.makedirs(odir, exist_ok=True)
    jobs = [
        joblib.delayed(_reformat_one)(fname, odir)
        for fname in fnames
    ]
    with joblib.Parallel(n_jobs=n_jobs, verbose=100) as par:
        par(jobs)


fnames = glob.glob(
    "/gpfs02/astro/desdata/esheldon/lensing/"
    "des-lensing/y6patches/patches-v5b/*.fits"
)
_reformat(fnames, "desdmv5a_cutsv5_patchesv5b")
