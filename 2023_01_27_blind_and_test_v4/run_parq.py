import fitsio
import glob
from des_y6utils.mdet import make_mdet_cuts
from esutil.pbar import PBar
import fastparquet
import pandas as pd
import numpy as np
import joblib
import gc


def _read_and_mask(fname):
    d = fitsio.read(fname)
    msk = make_mdet_cuts(d, "3")
    d = d[msk]
    return d


def main():
    fnames = glob.glob("blinded_data/*.fits")

    pq_fname = "metadetect_desdmv4_cutsv3.parq"
    first = True
    num_done = 0
    num_obj = 0

    loc = 0
    chunk_size = 10
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
            )
            first = False
            del _d
            gc.collect()


if __name__ == "__main__":
    main()