import sys
import os
import fitsio
import subprocess
import numpy as np
import glob
import time
from concurrent.futures import as_completed
from esutil.pbar import PBar
from mattspy import BNLCondorExecutor


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
    os.system("mkdir -p ./data")
    for _ in range(10):
        try:
            mfiles = _download_tile(tilename, cwd)
            break
        except Exception:
            time.sleep(600)
            mfiles = None
            pass

    if mfiles is None:
        raise RuntimeError("Could not download files for tile %s" % tilename)
    elif not isinstance(mfiles, list):
        raise RuntimeError("Only found %d files for tile %s" % (mfiles, tilename))

    if tmpdir is None:
        tmpdir = os.environ["TMPDIR"]

    cmd = """\
run-metadetect-on-slices \
  --config=%s/metadetect-v5-wmom.yaml \
  --output-path=./mdet_data \
  --seed=%d \
  --use-tmpdir \
  --tmpdir=%s \
  --log-level=INFO \
  --n-jobs=1 \
  --band-names=griz %s %s %s %s""" % (
        cwd, seed, tmpdir, mfiles[0], mfiles[1], mfiles[2], mfiles[3]
    )
    subprocess.run(cmd, shell=True, check=True)

    fnames = glob.glob("./mdet_data/%s*" % tilename)
    for fname in fnames:
        pth1 = os.path.realpath(os.path.abspath(fname))
        pth2 = os.path.realpath(os.path.abspath(
            os.path.join(opth, os.path.basename(fname)))
        )
        if pth1 != pth2:
            subprocess.run(
                "mv %s %s" % (pth1, pth2),
                shell=True,
                check=True,
            )


cwd = os.path.abspath(os.path.realpath(os.cwd()))
conda_env = "des-y6-final-v1"
seed = 100
os.system("mkdir -p ./mdet_data")
opth = os.path.abspath("./mdet_data")

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
    if sys.argv[1] == "download":
        d = fitsio.read(
            os.path.join(cwd, "fnames.fits"),
            lower=True,
        )
        tnames = sorted(list(set([
            d["filename"][i].split("_")[0]
            for i in range(d.shape[0])
        ])))
        import joblib

        jobs = []
        for tilename in tnames:
            jobs.append(joblib.delayed(_download_tile)(tilename))
        with joblib.Parallel(n_jobs=16, verbose=100) as par:
            par(jobs)

        sys.exit(0)
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
    tnames = np.array([
        d["filename"][i].split("_")[0]
        for i in range(d.shape[0])
    ])
    with BNLCondorExecutor(conda_env, debug=True, mem=4) as exec:
        futs = []
        nsub = 0
        for tilename, seed in zip(tnames, seeds):
            if (
                np.sum(tnames == tilename)
                and len(glob.glob("%s/%s*.fits.fz" % (opth, tilename))) == 0
            ):
                nsub += 1
                futs.append(
                    exec.submit(_run_tile, tilename, seed, opth, tmpdir, cwd)
                )
            if nsub % 32 == 0:
                time.sleep(600)
                nsub = 0

        for fut in PBar(as_completed(futs), total=len(futs), desc="running mdet"):
            try:
                fut.result()
            except Exception as e:
                print(repr(e), flush=True)
