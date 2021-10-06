import glob
import joblib
import fitsio
import os
import numpy as np
import meds
import tqdm

from meds.defaults import BMASK_EDGE

BINS = np.linspace(-20, 20, 41) + 0.5
BANDS = ["g", "r", "i", "z", "Y"]
BCEN = (BINS[:-1] + BINS[1:])/2


def _convert_to_index(row, col, dbox=100, edge=50):
    xind = (col.astype(int) - edge)//dbox
    yind = (row.astype(int) - edge)//dbox
    return xind + 99*yind


def _compute_hist_for_tile_band(tname, band):
    pizza_name = (
        "./pizza_meds/%s/p01/pizza-cutter/"
        "%s_r5191p01_%s_pizza-cutter-slices.fits.fz" % (
            tname,
            tname,
            band,
        )
    )
    stamp_name = glob.glob("./meds/%s_*_%s_meds-Y6A1.fits.fz" % (tname, band))[0]

    if (not os.path.exists(pizza_name)) or (not os.path.exists(stamp_name)):
        return None

    with meds.MEDS(pizza_name) as m, meds.MEDS(stamp_name) as mobj:
        pizza_inds = _convert_to_index(mobj["orig_row"][:, 0], mobj["orig_col"][:, 0])
        assert np.array_equal(
            _convert_to_index(m["orig_row"][:, 0], m["orig_col"][:, 0]),
            np.arange(len(m["orig_col"][:, 0]), dtype=int),
        )

        dnepoch = []
        nepoch = []
        for obj_ind, pizza_ind in tqdm.tqdm(
            enumerate(pizza_inds), total=len(pizza_inds)
        ):
            if pizza_ind < 0 or pizza_ind >= 9801 or m["nepoch"][pizza_ind] <= 0:
                continue

            nepoch_obj = 0
            for msk_ind in range(1, mobj["ncutout"][obj_ind]):
                msk = mobj.get_cutout(obj_ind, msk_ind, type="bmask")
                if not np.any(msk & BMASK_EDGE):
                    nepoch_obj += 1
            dnepoch.append(m["nepoch"][pizza_ind] - nepoch_obj)
            nepoch.append(nepoch_obj)

        print(
            "tile: %s\nband: %s\n\tpizza: %s %s\n\tmeds: %s %s\n\tdiff: %s %s" % (
                tname, band,
                np.mean(np.array(nepoch) + np.array(dnepoch)),
                np.std(np.array(nepoch) + np.array(dnepoch)),
                np.mean(nepoch), np.std(nepoch),
                np.mean(dnepoch), np.std(dnepoch),
            ),
            flush=True,
        )


tiles = list(set([os.path.basename(f).split("_")[0] for f in glob.glob("./meds/*")]))
assert len(tiles) == 100

os.system("mkdir -p hdata")

jobs = []
totd = []
for i, tile in enumerate(tiles):
    for band in BANDS:
        jobs.append(joblib.delayed(_compute_hist_for_tile_band)(tile, band))

with joblib.Parallel(n_jobs=5, backend='loky', verbose=100) as para:
    para(jobs)
