import os
import tempfile

import h5py
import joblib
import mattspy
import numpy as np
import treecorr
import tqdm


def _measure_for_tomo_ind(tomo_ind, wnum, opt, bin_edges, bin_ind):

    with h5py.File(
        "/astro/u/beckermr/workarea/des-y6-analysis/"
        "2025_06_04_double_check_cats/metadetect_2024-11-07.hdf5", "r"
    ) as fp:
        ra = fp["noshear"][f"tomo_bin_{tomo_ind}"]["ra"][:]
        dec = fp["noshear"][f"tomo_bin_{tomo_ind}"]["dec"][:]
        w = fp["noshear"][f"tomo_bin_{tomo_ind}"]["w"][:]

    if wnum == 1:
        wpart = "w"
        cat = treecorr.Catalog(ra=ra, dec=dec, k=np.ones_like(w), w=w, ra_units='deg', dec_units='deg')
    else:
        wpart = "w2"
        cat = treecorr.Catalog(ra=ra, dec=dec, k=np.ones_like(w), w=w*w, ra_units='deg', dec_units='deg')

    fname = f"{wpart}sum_b{tomo_ind}_{opt}_ab{bin_ind}.txt"
    if not os.path.exists(fname):
        kk = treecorr.KKCorrelation(
            # nbins=bin_edges.shape[0] - 1,
            # min_sep=bin_edges[0],
            # max_sep=bin_edges[-1],
            nbins=1,
            min_sep=bin_edges[bin_ind],
            max_sep=bin_edges[bin_ind+1],
            sep_units="arcmin",
            verbose=3,
            metric="Arc",
            bin_slop=0.01,
            angle_slop=0.01,
        )
        if opt == "cross":
            kk.process(cat1=cat, cat2=cat, num_threads=4)
        else:
            kk.process(cat, num_threads=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            _fname = os.path.join(tmpdir, fname)
            kk.write(_fname)
            with open(_fname) as fp:
                data = fp.read()

    return fname, data


bin_edges = np.array(
    [
        2.5       ,  3.14731353,  3.96223298,  4.98815579,  6.27971608,
        7.90569415,  9.95267926, 12.52968084, 15.77393361, 19.85820587,
        25.        , 31.47313529
    ]
)

with mattspy.BNLCondorParallel(
    n_jobs=4 * 2 * (bin_edges.shape[0]-1),
    cpus=1,
    mem=16,
    verbose=0,
) as exc:
    jobs = []
    for tomo_ind in range(4):
        for opt in ["cross"]:
            for wnum in [1, 2]:
                for bin_ind in range(bin_edges.shape[0]-1):
                    jobs.append(
                        joblib.delayed(_measure_for_tomo_ind)(
                            tomo_ind, wnum, opt, bin_edges, bin_ind
                        )
                    )
    futs = exc(jobs)
    for fut in tqdm.tqdm(futs, total=len(jobs)):
        fname, data = fut.result()
        with open(fname, "w") as fp:
            fp.write(data)
