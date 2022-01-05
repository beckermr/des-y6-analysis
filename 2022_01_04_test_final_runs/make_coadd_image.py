import subprocess
import glob
import concurrent.futures
import os
import tqdm


def _make_image(fnames, tname):
    for fname, band in zip(fnames, ["g", "r", "i"]):
        subprocess.run(
            "mkdir -p /data/beckermr/%s-%s" % (tname, band),
            shell=True,
            check=True,
        )
        subprocess.run(
            "make-coadd-image-from-slices %s "
            "--use-tmpdir --tmpdir=/data/beckermr/%s-%s "
            "--output-path=images/%s-%s.fits.fz" % (
                fname, tname, band, tname, band,
            ),
            check=True,
            shell=True,
        )
        return band

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


subprocess.run("mkdir -p images", check=True, shell=True)

mfiles = glob.glob(
    "data/"
    "OPS/multiepoch/Y6A2_PIZZACUTTER/*/*/*/pizza-cutter/"
    "*_pizza-cutter-slices.fits.fz"
)
tnames = set([os.path.basename(m).split("_")[0] for m in mfiles])

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futs = []

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
        ]

    futs.append(executor.submit(_make_image, mfiles, tname))

    for fut in concurrent.futures.as_completed(futs):
        print("done %s" % fut.result())

#     # --absscale 0.015 \
#     # --scales 1.0,1.0,1.3
