import os

import fitsio
import numpy as np


seed = 10
rng = np.random.RandomState(seed)
fname = "fake_stars.fits"

input_fname = os.path.expandvars(
    "${MEDS_DIR}/des-pizza-slices-y6-v14/DES0433-2332/sources-g/OPS_Taiga/cal/cat_tile_gaia/v1/DES0433-2332_GAIA_DR2_v1.fits"
)

din = fitsio.read(input_fname)

dtype_out = [
    ("ra", "f8"),
    ("dec", "f8"),
    ("gmag", "f8"),
    ("rmag", "f8"),
    ("imag", "f8"),
    ("zmag", "f8"),
]
dout = np.zeros(len(din), dtype=dtype_out)
dout["ra"] = din["RA"]
dout["dec"] = din["DEC"]
dout["gmag"] = rng.uniform(20, 25, size=len(din))
dout["rmag"] = rng.uniform(20, 25, size=len(din))
dout["imag"] = rng.uniform(20, 25, size=len(din))
dout["zmag"] = rng.uniform(20, 25, size=len(din))

fitsio.write(fname, dout, clobber=True)
