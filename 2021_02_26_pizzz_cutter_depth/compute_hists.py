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
    stamp_name = "./meds/%s_r4920p01_%s_meds-Y6A1.fits.fz" % (tname, band)

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
        h_pizza = np.histogram(m["nepoch"][pizza_ind], bins=BINS)[0]

        return h_pizza, h_nepoch, h_dnepoch


tiles = list(set([os.path.basename(f).split("_")[0] for f in glob.glob("./meds/*")]))
assert len(tiles) == 100

dtype = [
    ("pizza", "f4", (BINS.shape[0],)),
    ("stamp", "f4", (BINS.shape[0],)),
    ("diff", "f4", (BINS.shape[0],)),
    ("bin", "f4", (BINS.shape[0],)),
    ("tilename", "U20"),
    ("band", "U1"),
]

d = np.zeros(1, dtype=dtype)
res = _compute_hist_for_tile_band(tiles[0], BANDS[0])
d["band"][0] = BANDS[0]
d["tilename"][0] = tiles[0]
d["bin"][0] = BINS
d["pizza"][0] = res[0]
d["stamp"][0] = res[1]
d["diff"][0] = res[2]

fitsio.write("test.fits", d, clobber=True)
