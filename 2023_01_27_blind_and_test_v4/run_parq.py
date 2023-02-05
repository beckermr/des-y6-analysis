import fitsio
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from des_y6utils.mdet import make_mdet_cuts
from esutil.pbar import PBar
import fastparquet
import pandas as pd
import numpy as np


def _read_and_mask(fname):
    d = fitsio.read(fname)
    msk = make_mdet_cuts(d, "3")
    d = d[msk]
    return d


def main():
    fnames = glob.glob("blinded_data/*.fits")

    pq_fname = "mdet_desdmv4_cutsv3.parq"
    first = True
    num_done = 0
    num_obj = 0
    cats = []

    with ProcessPoolExecutor(max_workers=10) as exc:
        futs = [
            exc.submit(_read_and_mask, fname)
            for fname in PBar(fnames, desc="making jobs")
        ]
        for fut in PBar(
            as_completed(futs), total=len(futs), desc="appending catalogs"
        ):
            try:
                _d = fut.result()
            except Exception as e:
                print(e)
                _d = None

            if _d is not None:
                cats.append(_d)

            if len(cats) == 10:
                num_done += len(cats)
                _d = np.concatenate(cats, axis=0)
                cats = []
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
                # print(num_done, num_obj/1e6)

    if len(cats) > 0:
        num_done += len(cats)
        _d = np.concatenate(cats, axis=0)
        cats = []
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


if __name__ == "__main__":
    main()