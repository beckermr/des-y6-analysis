import glob
import joblib
import fitsio
import os
import numpy as np
import meds
import tqdm
import esutil

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

        h_dnepoch = np.histogram(dnepoch, bins=BINS)[0]
        h_nepoch = np.histogram(nepoch, bins=BINS)[0]
        h_pizza = np.histogram(np.array(nepoch) + np.array(dnepoch), bins=BINS)[0]

        return h_pizza, h_nepoch, h_dnepoch, tname, band


tiles = list(set([os.path.basename(f).split("_")[0] for f in glob.glob("./meds/*")]))
assert len(tiles) == 100

dtype = [
    ("pizza", "f4", (BINS.shape[0]-1,)),
    ("stamp", "f4", (BINS.shape[0]-1,)),
    ("diff", "f4", (BINS.shape[0]-1,)),
    ("bin", "f4", (BINS.shape[0]-1,)),
    ("tilename", "U20"),
    ("band", "U1"),
]

os.system("rm -rf test.fits")

jobs = []
totd = []
for i, tile in enumerate(tiles):
    for band in BANDS:
        jobs.append(joblib.delayed(_compute_hist_for_tile_band)(tile, band))

        with joblib.Parallel(n_jobs=5, backend='loky', verbose=100, timeout=240) as para:
            outputs = para(jobs)
        print("done with data processing", flush=True)

        jobs = []

        outputs = [o for o in outputs if o is not None]

        if len(outputs) > 0:
            d = np.zeros(len(outputs), dtype=dtype)
            for i, res in enumerate(outputs):
                d["band"][i] = res[4]
                d["tilename"][i] = res[3]
                d["bin"][i] = BCEN
                d["pizza"][i] = res[0]
                d["stamp"][i] = res[1]
                d["diff"][i] = res[2]

            totd.append(d)

            print("writing data", flush=True)
            fitsio.write("test.fits", esutil.numpy_util.combine_arrlist(totd), clobber=True)
