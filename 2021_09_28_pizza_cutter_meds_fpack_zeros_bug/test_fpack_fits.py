import subprocess
import numpy as np
import fitsio

subprocess.run(
    "rm -rf test.fits*",
    shell=True,
    check=True,
)

da = np.zeros(400000, dtype=[('blah', 'f8')])

d = np.zeros(400000, dtype=np.float32)
d[:] = np.random.uniform(size=400000) * 0.18
d[31:35] = 0
assert not np.any(d < 0)

fpack_header = {
    "FZQVALUE": 4,
    "FZTILE": "(13,1)",
    "FZALGOR": "RICE_1",
    # preserve zeros, don't dither them
    "FZQMETHD": "SUBTRACTIVE_DITHER_2",
}

with fitsio.FITS("test.fits", "rw", clobber=True) as fits:
    fits.create_image_hdu(
        img=None,
        dtype=np.float32,
        dims=d.shape,
        extname="base",
        header=fpack_header,
    )

    # also need to write the header...IDK why...
    fits["base"].write_keys(fpack_header, clean=False)
    fits["base"].write(d)

    # fits.create_image_hdu(
    #     img=None,
    #     dtype=np.float32,
    #     dims=d.shape,
    #     extname="base-nc",
    #     # header={"FZALGOR": "NONE"},
    # )
    # fits["base-nc"].write_keys({"FZALGOR": "NONE"}, clean=False)
    # fits["base-nc"].write(d)

print("fits header:\n", fitsio.read_header("test.fits"))


subprocess.run(
    "./cfitsio/fpack test.fits",
    shell=True,
    check=True,
)

# print("fits.fz header:\n", fitsio.read_header("test.fits.fz", ext="base-nc"))

print("fits.fz header:\n", fitsio.read_header("test.fits.fz", ext="base"))

dr = fitsio.read("test.fits.fz")

assert not np.any(dr[31:35] != 0)
assert not np.any(dr < 0), dr[dr < 0]
