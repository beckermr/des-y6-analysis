import sys
import numpy as np
import meds
import fitsio

print(fitsio.read_header(sys.argv[1], ext="mfrac_cutouts"))

m = meds.MEDS(sys.argv[1])

for i in range(len(m['ncutout'])):
    if m["ncutout"][i] <= 0:
        continue
    if np.any(m.get_cutout(i, 0, type="weight") < 0):
        print("%d: weight bad: " % (i), np.min(m.get_cutout(i, 0, type="weight")))
        break

    im = m.get_cutout(i, 0, type="mfrac")
    if np.any(im < 0):
        print(im.min(), im.max())
        print("%d: mfrac bad: " % (i), np.min(im))
