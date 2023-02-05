import fitsio
import glob
from des_y6utils.mdet import make_mdet_cuts
from esutil.pbar import PBar
import fastparquet
import pandas as pd
import numpy as np
import joblib
import gc
from numpy.lib.recfunctions import repack_fields


COLS = (
    'uid',
    'patch_num',
    'tilename',
    'slice_id',
    'mdet_step',
    'ra',
    'dec',
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
    'gauss_s2n',
    'gauss_g_1',
    'gauss_g_2',
    'gauss_g_cov_1_1',
    'gauss_g_cov_1_2',
    'gauss_g_cov_2_2',
    'gauss_T_err',
    'gauss_T_ratio',
    'gauss_psf_T',
    'pgauss_T_err',
    'pgauss_T',
    'pgauss_psf_T',
    'pgauss_band_flux_g',
    'pgauss_band_flux_r',
    'pgauss_band_flux_i',
    'pgauss_band_flux_z',
    'pgauss_band_flux_err_g',
    'pgauss_band_flux_err_r',
    'pgauss_band_flux_err_i',
    'pgauss_band_flux_err_z',
)


def _read_and_mask(fname):
    d = fitsio.read(fname)
    msk = make_mdet_cuts(d, "3")
    d = d[msk]
    return repack_fields(d[[col for col in COLS if col in d.dtype.names]])


def main():
    fnames = glob.glob(
        "/gpfs02/astro/desdata/esheldon/lensing/"
        "des-lensing/y6patches/patches/*.fits"
    )[0:3]

    pq_fname = "metadetect_desdmv4_cutsv3_jk"
    first = True
    num_done = 0
    num_obj = 0

    loc = 0
    chunk_size = 1
    chunks = len(fnames) // chunk_size
    chunks += 1

    with joblib.Parallel(n_jobs=chunk_size, verbose=0) as par:
        for chunk in PBar(range(chunks)):
            max_loc = min(loc + chunk_size, len(fnames))
            _fnames = fnames[loc:max_loc]
            jobs = [joblib.delayed(_read_and_mask)(fn) for fn in _fnames]
            _d = np.concatenate(par(jobs), axis=0)
            num_done += (max_loc-loc)
            num_obj += len(_d)
            _d = pd.DataFrame(_d)
            fastparquet.write(
                pq_fname, _d,
                has_nulls=False,
                write_index=False,
                fixed_text={"mdet_step": len("noshear")},
                compression="SNAPPY",
                append=False if first else True,
                row_group_offsets=1_000_000,
                file_scheme="drill",
                partition_on=["patch_num", "mdet_step"],
            )
            first = False
            del _d
            gc.collect()


if __name__ == "__main__":
    main()
