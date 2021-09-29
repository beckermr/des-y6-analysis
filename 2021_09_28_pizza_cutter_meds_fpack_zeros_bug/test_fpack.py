import subprocess
import numpy as np
import fitsio

subprocess.run(
    "rm -rf test.fits*",
    shell=True,
    check=True,
)

da = np.zeros(20, dtype=[('blah', 'f8')])

d = np.zeros(20, dtype=np.float32)
d[5:] = np.random.uniform(size=20-5) * 0.18

fpack_header = {
    "FZQVALUE": 4,
    "FZTILE": "(13,1)",
    "FZALGOR": "RICE_1",
    # preserve zeros, don't dither them
    "FZQMETHD": "SUBTRACTIVE_DITHER_2",
}


fitsio.write("test.fits", da, clobber=True, header=fpack_header, extname="base-arr")

fitsio.write("test.fits", d, header=fpack_header, extname="base")

for i in range(20):
    if i % 2 == 0:
        dw = d.astype(np.int32)
    else:
        dw = d
    fitsio.write("test.fits", dw, header=fpack_header, extname="base%d" % i)


subprocess.run(
    "./cfitsio/fpack test.fits",
    shell=True,
    check=True,
)

for i in range(20):
    if i % 2 == 0:
        continue

    dr = fitsio.read("test.fits.fz", ext="base%d" % i)

    assert not np.any(dr[0:5] != 0)
    assert np.any(dr != 0)
