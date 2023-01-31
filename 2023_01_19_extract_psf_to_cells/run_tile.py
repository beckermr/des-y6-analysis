import sys
import os
import fitsio
import subprocess
import numpy as np
from esutil.pbar import PBar
from mattspy import BNLCondorParallel
import esutil
import joblib


def _download_tile(tilename, cwd):
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
    for band in ["g", "r", "i", "z"]:
        msk = d["band"] == band
        if np.any(msk):
            _d = d[msk]
            for i in range(len(_d)):
                fname = os.path.join(d["path"][msk][i], d["filename"][msk][i])
                if not os.path.exists("./data/%s" % os.path.basename(fname)):
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


def _extract_expnum_ccdnum(ii):
    expnums = np.zeros(len(ii), dtype=int) + -1
    ccdnums = np.zeros(len(ii), dtype=int) + -1
    for i in range(len(ii)):
        ip = ii["image_path"][i]
        if len(ip) > 0:
            expnums[i] = int(os.path.basename(ip).split("_")[0][1:])
            ccdnums[i] = int(os.path.basename(ip).split("_")[2][1:])
    return expnums, ccdnums


def _process_file(fname, tname, band):
    ei = fitsio.read(fname, ext="epochs_info")
    ii = fitsio.read(fname, ext="image_info")
    ei = esutil.numpy_util.add_fields(
        ei,
        [
            ("tilename", "U12"),
            ("band", "U1"),
            ("expnum", "i8"),
            ("ccdnum", "i4"),
            ("image_flags", "i4"),
        ],
    )
    ei["tilename"] = tname
    ei["band"] = band

    expnums, ccdnums = _extract_expnum_ccdnum(ii)
    ei["expnum"] = expnums[ei["image_id"]]
    ei["ccdnum"] = ccdnums[ei["image_id"]]
    ei["image_flags"] = ii["image_flags"][ei["image_id"]]
    assert np.all(ei["ccdnum"] <= 62)
    assert np.array_equal(ii["image_id"], np.arange(len(ii)))

    try:
        os.remove(fname)
    except Exception:
        pass

    return ei, ii


def _process_tile(tname, cwd):
    fnames = _download_tile(tname, cwd)
    for fname in fnames:
        band = os.path.basename(fname).split("_")[2]
        tname = os.path.basename(fname).split("_")[0]
        assert band in ["g", "r", "i", "z"]
        ei, _ = _process_file(fname, tname, band)

        ofname = os.path.join(
            cwd,
            "epochs_info",
            band,
            os.path.basename(fname)[:-len("pizza-cutter-slices.fits.fz")]
            + "epochs-info.fits",
        )

        fitsio.write(ofname, ei, clobber=True)


def main():
    cwd = os.path.abspath(os.path.realpath(os.getcwd()))

    tnames = np.unique(np.array([
        fn.split("_")[0]
        for fn in fitsio.read("fnames.fits")["FILENAME"]
    ]))

    if len(sys.argv) == 2:
        tnames = tnames[:int(sys.argv[1])]

    for band in "griz":
        os.makedirs("./epochs_info/" + band, exist_ok=True)
    os.system("mkdir -p data")

    with BNLCondorParallel(verbose=0, mem=2, cpus=1, n_jobs=40) as exc:
        jobs = []
        for tilename in PBar(tnames, total=len(tnames), desc="making jobs"):
            jobs.append(
                joblib.delayed(_process_tile)(
                    tilename, cwd,
                )
            )

        for res in PBar(exc(jobs), total=len(jobs), desc="processing tiles"):
            try:
                res.result()
            except Exception as e:
                print("ERROR: " + repr(e), flush=True)


if __name__ == "__main__":
    main()
