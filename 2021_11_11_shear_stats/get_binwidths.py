import glob
import fitsio
import tqdm
import numpy as np

fnames = glob.glob("./data/**/*.fits.fz", recursive=True)


gvals = []
for fname in tqdm.tqdm(fnames):
    d = fitsio.read(fname)
    msk = (
        (d["mask_flags"] == 0)
        & (d["flags"] == 0)
        & (d["mdet_s2n"] > 10)
        & (d["mdet_T_ratio"] > 1.2)
        & (d["mfrac"] < 0.1)
    )
    gvals.append(d["psfrec_g"][msk, :].ravel())
    be = np.quantile(np.hstack(gvals), np.linspace(0, 1, 16))
    print(be)


fitsio.write(
    "bin_edges.fits",
    be,
    clobber=True,
)
