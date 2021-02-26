import glob
import fitsio
import tqdm

import esutil as eu


fnames = glob.glob("./hdata/*.fits")

d = []
for fname in tqdm.tqdm(fnames):
    d.append(fitsio.read(fname))

dtot = eu.numpy_util.combine_arrlist(d)

fitsio.write("hdata.fits", dtot, clobber=True)
