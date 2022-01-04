import sys
import subprocess
import glob
import concurrent.futures
import os
import tqdm

mfiles = glob.glob(
    "data/"
    "OPS/multiepoch/Y6A2_PIZZACUTTER/*/*/*/pizza-cutter/"
    "*_pizza-cutter-slices.fits.fz"
)
tnames = set([os.path.basename(m).split("_")[0] for m in mfiles])

for tname in tqdm.tqdm(tnames):

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
            "mkdir -p /data/beckermr/%s" % tname,
            shell=True,
            check=True,
        )
        subprocess.run(
            "make-coadd-image-from-slices %s "
            "--use-tmpdir --tmpdir=/data/beckermr/%s "
            "--output-path=images/%s-%s.fits.fz" % (
                fname, tname, tname, band,
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

    subprocess.run(
        "rm -f images/%s-*.fits.fz" % tname,
        shell=True,
        check=True,
    )

#     # --absscale 0.015 \
#     # --scales 1.0,1.0,1.3
