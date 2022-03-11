import subprocess
import glob
import concurrent.futures
import os
import tqdm
import sys
import fitsio
import numpy as np


def _download_tile(tilename, cwd="."):
    os.system("mkdir -p data")

    d = fitsio.read(
        os.path.join(cwd, "fnames.fits"),
        lower=True,
    )
    tnames = np.array([
        d["filename"][i].split("_")[0]
        for i in range(d.shape[0])
    ])
    msk = tnames == tilename
    if np.sum(msk) != 4:
        return np.sum(msk)

    d = d[msk]
    mfiles = []
    for band in ["g", "r", "i"]:
        msk = d["band"] == band
        if np.any(msk):
            _d = d[msk]
            for i in range(len(_d)):
                fname = os.path.join(d["path"][msk][i], d["filename"][msk][i])
                cmd = """\
        rsync \
                -av \
                --password-file $DES_RSYNC_PASSFILE \
                ${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/%s \
                ./data/%s
        """ % (fname, os.path.basename(fname))
                subprocess.run(cmd, shell=True, check=True)
            mfiles.append("./data/%s" % os.path.basename(fname))

    return mfiles


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


def main():
    tilename = sys.argv[1]

    _download_tile(tilename, cwd=".")

    subprocess.run("mkdir -p images", check=True, shell=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futs = []

        mfiles = glob.glob(
            "data/"
            "OPS/multiepoch/Y6A2_PIZZACUTTER/*/%s/*/pizza-cutter/"
            "%s_r*_*_pizza-cutter-slices.fits.fz" % (
                tilename, tilename,
            )
        )
        mfiles = sorted(mfiles)
        mfiles = [
            mfiles[0],
            mfiles[2],
            mfiles[1],
        ]

        futs.append(executor.submit(_make_image, mfiles, tilename))

        for fut in tqdm.tqdm(
            concurrent.futures.as_completed(futs), total=len(futs)
        ):
            print("done %s" % fut.result())


if __name__ == "__main__":
    main()
