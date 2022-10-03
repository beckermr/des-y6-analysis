import sys
import os
import fitsio
import subprocess
import numpy as np
import glob
import time
import random
import joblib
from esutil.pbar import PBar
from mattspy import BNLCondorParallel


def _download_tile(tilename, cwd):
    d = fitsio.read(
        os.path.join(cwd, "fnames.fits"),
        lower=True,
    )
    tnames = np.array([
        d["filename"][i].split("_")[0]
        for i in range(d.shape[0])
    ])
    msk = tnames == tilename
    if np.sum(msk) != 1:
        return np.sum(msk)

    d = d[msk]
    mfiles = []
    for band in ["r"]:
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


def _run_tile(tilename, seed, opth, tmpdir, cwd):
    for _ in range(10):
        try:
            mfiles = _download_tile(tilename, cwd)
            break
        except Exception:
            time.sleep(600 + random.randint(-100, 100))
            mfiles = None
            pass

    if mfiles is None:
        raise RuntimeError("Could not download files for tile %s" % tilename)
    elif not isinstance(mfiles, list):
        raise RuntimeError("Only found %d files for tile %s" % (mfiles, tilename))

    for mfile in mfiles:
        try:
            d = fitsio.read(mfile, ext="object_data")
            ofname = os.path.join(opth, "%s_object_data.fits" % tilename)
            fitsio.write(ofname, d, clobber=True)
        except Exception as e:
            raise e
        else:
            os.system("rm -f %s" % mfile)


cwd = os.path.abspath(os.path.realpath(os.getcwd()))
seed = 100
os.system("mkdir -p ./data")
os.system("mkdir -p ./pz_data")
opth = os.path.abspath("./pz_data")

if len(sys.argv) == 1:
    d = fitsio.read(
        os.path.join(cwd, "fnames.fits"),
        lower=True,
    )
    tnames = sorted(list(set([
        d["filename"][i].split("_")[0]
        for i in range(d.shape[0])
    ])))
    rng = np.random.RandomState(seed=seed)
    seeds = rng.randint(low=1, high=2**29, size=len(tnames))
    tmpdir = None

else:
    tnames = [sys.argv[1]]
    tmpdir = "/data/beckermr/tmp/" + tnames[0] + "_mdet"
    os.system("mkdir -p " + tmpdir)

if len(tnames) == 1:
    _run_tile(tnames[0], seed, opth, tmpdir, cwd)
else:
    d = fitsio.read(
        os.path.join(cwd, "fnames.fits"),
        lower=True,
    )
    all_tnames = np.array([
        d["filename"][i].split("_")[0]
        for i in range(d.shape[0])
    ])
    with BNLCondorParallel(verbose=0, mem=2) as exec:
        jobs = []
        for tilename, seed in PBar(
            zip(tnames, seeds), total=len(tnames), desc="making jobs"
        ):
            if (
                np.sum(all_tnames == tilename) == 1
                and len(glob.glob("%s/%s*.fits" % (opth, tilename))) == 0
            ):
                jobs.append(
                    joblib.delayed(_run_tile)(tilename, seed, opth, tmpdir, cwd)
                )

        for res in PBar(exec(jobs), total=len(jobs), desc="running mdet"):
            try:
                res.result()
            except Exception as e:
                print("ERROR: " + repr(e), flush=True)
