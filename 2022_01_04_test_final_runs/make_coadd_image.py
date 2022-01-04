import sys
import subprocess
import glob
import concurrent.futures

tname = sys.argv[1]

mfiles = glob.glob(
    "data/"
    "OPS/multiepoch/Y6A2_PIZZACUTTER/*/%s/*/pizza-cutter/"
    "%s_r*_*_pizza-cutter-slices.fits.fz" % (
        tname, tname,
    )
)
mfiles = sorted(mfiles)
mfiles = [
    mfiles[0],
    mfiles[2],
    mfiles[1],
    mfiles[3]
]

subprocess.run("mkdir -p images", check=True, shell=True)


def _render(fname, tname, band):
    subprocess.run(
        "make-coadd-image-from-slices %s --output-path=images/%s-%s.fits.fz" % (
            fname, tname, band,
        ),
        check=True,
        shell=True,
    )
    return band


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futs = []
    for band, fname in zip(["g", "r", "i", "z"], mfiles):
        assert fname.endswith("%s_pizza-cutter-slices.fits.fz" % band)
        futs.append(executor.submit(_render, fname, tname, band))

    for fut in concurrent.futures.as_completed(futs):
        print("done %s" % fut.result())


subprocess.run(
    """\
des-make-image-fromfiles \
    images/%s-coadd-gri.jpg \
    images/%s-g.fits.fz \
    images/%s-r.fits.fz \
    images/%s-i.fits.fz \
""" % (tname, tname, tname, tname),
    check=True,
    shell=True,
)

#     # --absscale 0.015 \
#     # --scales 1.0,1.0,1.3
