import fitsio
import numpy as np
import tqdm
import glob
import joblib


# see here https://en.wikipedia.org/wiki/
#          Algorithms_for_calculating_variance#Weighted_incremental_algorithm
def _online_update(e, e_err, n, n2, _e, _n, row, col):
    e_old = e[row, col].copy()
    e[row, col] = e_old + (_n / n[row, col]) * (_e - e_old)
    e_err[row, col] = e_err[row, col] + _n * (_e - e_old) * (_e - e[row, col])
    return e, e_err


def _reduce_per_ccd(fnames):
    data = {}
    for i in range(62):
        ccd = i + 1
        data[ccd] = dict(
            n=np.zeros((32, 16)),
            n2=np.zeros((32, 16)),
            e1=np.zeros((32, 16)),
            e1_err=np.zeros((32, 16)),
            e2=np.zeros((32, 16)),
            e2_err=np.zeros((32, 16)),
        )

    for fname in tqdm.tqdm(fnames, ncols=79, desc="per ccd"):
        print("\n", end="", flush=True)
        d = fitsio.read(fname)
        d = d[d["n"] > 0]

        for i in tqdm.trange(62, ncols=79, desc="reducing ccds"):
            print("\n", end="", flush=True)
            ccd = i + 1
            ccd_msk = d["ccdnum"] == ccd
            _d = d[ccd_msk]
            for row in range(32):
                for col in range(16):
                    msk = (_d["row_bin"] == row) & (_d["col_bin"] == col)
                    if np.any(msk):
                        _n = np.sum(_d["n"][msk])
                        data[ccd]["n"][row, col] += _n
                        data[ccd]["n2"][row, col] += _n**2

                        _e = np.mean(_d["e1"][msk])
                        e1, e1_err = _online_update(
                            data[ccd]["e1"], data[ccd]["e1_err"],
                            data[ccd]["n"], data[ccd]["n2"],
                            _e, _n, row, col,
                        )
                        data[ccd]["e1"] = e1
                        data[ccd]["e1_err"] = e1_err

                        _e = np.mean(_d["e2"][msk])
                        e2, e2_err = _online_update(
                            data[ccd]["e2"], data[ccd]["e2_err"],
                            data[ccd]["n"], data[ccd]["n2"],
                            _e, _n, row, col,
                        )
                        data[ccd]["e2"] = e2
                        data[ccd]["e2_err"] = e2_err

    for i in range(62):
        ccd = i + 1
        data[ccd]["e1_err"] = np.sqrt(data[ccd]["e1_err"] / (data[ccd]["n"] - 1))
        data[ccd]["e2_err"] = np.sqrt(data[ccd]["e2_err"] / (data[ccd]["n"] - 1))

        fitsio.write(
            "cte_data_all_ccd%02d.fits" % ccd, data[ccd]["e1"], extname="e1",
            clobber=True,
        )
        fitsio.write(
            "cte_data_all_ccd%02d.fits" % ccd, data[ccd]["e1_err"], extname="e1_err")
        fitsio.write(
            "cte_data_all_ccd%02d.fits" % ccd, data[ccd]["e2"], extname="e2")
        fitsio.write(
            "cte_data_all_ccd%02d.fits" % ccd, data[ccd]["e2_err"], extname="e2_err")


def _reduce_per_ccd_all(fnames):

    n = np.zeros((32, 16))
    n2 = np.zeros((32, 16))
    e1 = np.zeros((32, 16))
    e1_err = np.zeros((32, 16))
    e2 = np.zeros((32, 16))
    e2_err = np.zeros((32, 16))

    for fname in tqdm.tqdm(fnames, ncols=79, desc="all CCDs"):
        print("\n", end="", flush=True)
        d = fitsio.read(fname)

        ccd_msk = (d["n"] > 0)
        d = d[ccd_msk]
        for row in range(32):
            for col in range(16):
                msk = (d["row_bin"] == row) & (d["col_bin"] == col)
                if np.any(msk):
                    _n = np.sum(d["n"][msk])
                    n[row, col] += _n
                    n2[row, col] += _n**2

                    _e = np.mean(d["e1"][msk])
                    e1, e1_err = _online_update(e1, e1_err, n, n2, _e, _n, row, col)

                    _e = np.mean(d["e2"][msk])
                    e2, e2_err = _online_update(e2, e2_err, n, n2, _e, _n, row, col)

    e1_err = np.sqrt(e1_err / (n - 1))
    e2_err = np.sqrt(e2_err / (n - 1))

    fitsio.write("cte_data_all_ccd.fits", e1, extname="e1", clobber=True)
    fitsio.write("cte_data_all_ccd.fits", e1_err, extname="e1_err")
    fitsio.write("cte_data_all_ccd.fits", e2, extname="e2")
    fitsio.write("cte_data_all_ccd.fits", e2_err, extname="e2_err")


def _online_update_one(e, e_err, n, _e, _err, _n, ind):
    delta = e[ind] - _e
    e[ind] = (_e*_n + e[ind]*(n[ind] - _n)) / n[ind]
    e_err[ind] = e_err[ind] + _err + delta**2 * (n[ind] - _n) * _n / n[ind]
    return e, e_err


def _reduce_rows_cols(fnames, shape, col, desc, loc_col, oname):

    n = np.zeros(shape)
    e1 = np.zeros(shape)
    e1_err = np.zeros(shape)
    e2 = np.zeros(shape)
    e2_err = np.zeros(shape)
    loc = np.zeros(shape)

    for fname in tqdm.tqdm(fnames, ncols=79, desc=desc):
        print("\n", end="", flush=True)
        d = fitsio.read(fname)

        msk = (d["n"] > 0)
        d = d[msk]
        for b in range(shape):
            msk = (d[col] == b)
            if np.any(msk):
                _n = np.sum(d["n"][msk])
                n[b] += _n

                _e = np.mean(d["e1"][msk])
                _err = np.sum((d["e1"][msk] - _e)**2)
                e1, e1_err = _online_update_one(e1, e1_err, n, _e, _err, _n, b)

                _e = np.mean(d["e2"][msk])
                _err = np.sum((d["e2"][msk] - _e)**2)
                e2, e2_err = _online_update_one(e2, e2_err, n, _e, _err, _n, b)

                loc[b] += np.sum(d[loc_col][msk])

    e1_err = np.sqrt(e1_err / (n - 1)) / np.sqrt(n)
    e2_err = np.sqrt(e2_err / (n - 1)) / np.sqrt(n)

    loc = loc / n

    fitsio.write(oname, e1, extname="e1", clobber=True)
    fitsio.write(oname, e1_err, extname="e1_err")
    fitsio.write(oname, e2, extname="e2")
    fitsio.write(oname, e2_err, extname="e2_err")
    fitsio.write(oname, loc, extname=loc_col)


def main():
    fnames = glob.glob("cte_data_all_*.fits")
    fnames = [
        f
        for f in fnames
        if f.split(".")[0].split("_")[-1].isdigit()
    ]

    jobs = [
        joblib.delayed(_reduce_rows_cols)(
            fnames, 16, "col_bin", "reducing cols", "col",
            "cte_data_all_col.fits"
        ),
        # joblib.delayed(_reduce_per_ccd)(fnames),
        joblib.delayed(_reduce_rows_cols)(
            fnames, 32, "row_bin", "reducing rows", "row",
            "cte_data_all_row.fits"
        ),
        # joblib.delayed(_reduce_per_ccd_all)(fnames),
    ]

    with joblib.Parallel(n_jobs=2, verbose=100) as par:
        par(jobs)


if __name__ == "__main__":
    main()
