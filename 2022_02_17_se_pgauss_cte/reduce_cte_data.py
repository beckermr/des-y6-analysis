import fitsio
import numpy as np
import tqdm
import glob
import joblib


# see here https://en.wikipedia.org/wiki/
#          Algorithms_for_calculating_variance#Weighted_incremental_algorithm
def _online_update(e, e_err, n, n2, _e, _n, row, col):
    n[row, col] += _n
    n2[row, col] += _n**2

    e_old = e[row, col].copy()
    e[row, col] = e_old + (_n / n[row, col]) * (_e - e_old)
    e_err[row, col] = e_err[row, col] + _n * (_e - e_old) * (_e - e[row, col])
    return e, e_err, n, n2


def _reduce_per_ccd(fnames, ccd):

    ns = np.zeros((32, 16))
    n = np.zeros((32, 16))
    n2 = np.zeros((32, 16))
    e1 = np.zeros((32, 16))
    e1_err = np.zeros((32, 16))
    e2 = np.zeros((32, 16))
    e2_err = np.zeros((32, 16))

    for fname in tqdm.tqdm(fnames, ncols=79):
        d = fitsio.read(fname)

        ccd_msk = (d["n"] > 0) & (d["ccdnum"] == ccd)
        d = d[ccd_msk]
        for row in tqdm.trange(32, ncols=79):
            for col in range(16):
                msk = (d["row_bin"] == row) & (d["col_bin"] == col)
                _n = np.sum(d["n"][msk])

                _e = np.mean(d["e1"][msk])
                e1, e1_err, n, n2 = _online_update(e1, e1_err, n, n2, _e, _n, row, col)

                _e = np.mean(d["e2"][msk])
                e1, e1_err, n, n2 = _online_update(e2, e2_err, n, n2, _e, _n, row, col)

                ns[row, col] = np.sum(msk)

    e1_err = np.sqrt(e1_err / (n - 1)) / np.sqrt(ns)
    e2_err = np.sqrt(e2_err / (n - 1)) / np.sqrt(ns)


def _online_update_one(e, e_err, n, n2, _e, _n, ind):
    e_old = e[ind].copy()
    e[ind] = e_old + (_n / n[ind]) * (_e - e_old)
    e_err[ind] = e_err[ind] + _n * (_e - e_old) * (_e - e[ind])
    return e, e_err, n, n2


def _reduce_rows_cols(fnames, shape, col, desc, loc_col):

    ns = np.zeros(shape)
    n = np.zeros(shape)
    n2 = np.zeros(shape)
    e1 = np.zeros(shape)
    e1_err = np.zeros(shape)
    e2 = np.zeros(shape)
    e2_err = np.zeros(shape)
    loc = np.zeros(shape)

    for fname in tqdm.tqdm(fnames, ncols=79, desc=desc):
        d = fitsio.read(fname)

        msk = (d["n"] > 0)
        d = d[msk]
        for b in range(shape):
            msk = (d[col] == b)

            _n = np.sum(d["n"][msk])
            n[b] += _n
            n2[b] += _n**2

            _e = np.mean(d["e1"][msk])
            e1, e1_err, n, n2 = _online_update_one(e1, e1_err, n, n2, _e, _n, b)

            _e = np.mean(d["e2"][msk])
            e1, e1_err, n, n2 = _online_update_one(e2, e2_err, n, n2, _e, _n, b)

            loc[b] += np.sum(d[loc_col][msk])

            ns[b] = np.sum(msk)

            print(loc)

    e1_err = np.sqrt(e1_err / (n - 1)) * np.sqrt(ns)
    e2_err = np.sqrt(e2_err / (n - 1)) * np.sqrt(ns)
    loc /= n

    return e1, e1_err, e2, e2_err, loc


def main():
    fnames = glob.glob("cte_data_all_*.fits")
    fnames = [
        f
        for f in fnames
        if f.split(".")[0].split("_")[-1].isdigit()
    ]

    e1, e1_err, e2, e2_err, row = _reduce_rows_cols(
        fnames, 16, "col_bin", "reducing cols", "col"
    )
    fitsio.write("cte_data_all_col.fits", e1, extname="e1", clobber=True)
    fitsio.write("cte_data_all_col.fits", e1_err, extname="e1_err")
    fitsio.write("cte_data_all_col.fits", e2, extname="e2")
    fitsio.write("cte_data_all_col.fits", e2_err, extname="e2_err")
    fitsio.write("cte_data_all_col.fits", row, extname="col")


if __name__ == "__main__":
    main()
