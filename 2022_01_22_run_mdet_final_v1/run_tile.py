import sys
import os
import fitsio
import subprocess
import numpy as np
import glob
from concurrent.futures import as_completed
from esutil.pbar import PBar
from condor_exec import CondorExecutor


def _run_tile(tilename, seed, opth, tmpdir):
    os.system("mkdir -p data")

    d = fitsio.read(
        "/astro/u/beckermr/workarea/des-y6-analysis/"
        "2022_01_22_run_mdet_final_v1/fnames.fits",
        lower=True,
    )
    tnames = np.array([
        d["filename"][i].split("_")[0]
        for i in range(d.shape[0])
    ])
    msk = tnames == tilename
    d = d[msk]
    mfiles = []
    for band in ["g", "r", "i", "z"]:
        msk = d["band"] == band
        assert np.sum(msk) == 1
        fname = os.path.join(d["path"][msk][0], d["filename"][msk][0])
        cmd = """\
rsync \
        -av \
        --password-file $DES_RSYNC_PASSFILE \
        ${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/%s \
        ./data/%s
""" % (fname, os.path.basename(fname))
        subprocess.run(cmd, shell=True, check=True)
        mfiles.append("./data/%s" % os.path.basename(fname))

    if tmpdir is None:
        tmpdir = os.environ["TMPDIR"]

    cmd = """\
run-metadetect-on-slices \
  --config=/astro/u/beckermr/workarea/des-y6-analysis/\
2022_01_22_run_mdet_final_v1/metadetect-v5.yaml \
  --output-path=./mdet_data \
  --seed=%d \
  --use-tmpdir \
  --tmpdir=%s \
  --log-level=INFO \
  --n-jobs=1 \
  --range=0:10 \
  --band-names=griz %s %s %s %s""" % (
        seed, tmpdir, mfiles[0], mfiles[1], mfiles[2], mfiles[3]
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


conda_env = "des-y6-final-v1"
seed = 100
os.system("mkdir -p ./mdet_data")
opth = os.path.abspath("./mdet_data")

if len(sys.argv) == 1:
    d = fitsio.read(
        "/astro/u/beckermr/workarea/des-y6-analysis/"
        "2022_01_22_run_mdet_final_v1/fnames.fits",
        lower=True,
    )
    tnames = sorted(list(set([
        d["filename"][i].split("_")[0]
        for i in range(d.shape[0])
    ])))
    tnames = tnames[0:10]
    rng = np.random.RandomState(seed=seed)
    seeds = rng.randint(low=1, high=2**29, size=len(tnames))
    tmpdir = None

else:
    tnames = [sys.argv[1]]
    tmpdir = "/data/beckermr/tmp/" + tnames[0] + "_mdet"
    os.system("mkdir -p " + tmpdir)

if len(tnames) == 1:
    _run_tile(tnames[0], seed, opth, tmpdir)
else:
    with CondorExecutor(conda_env=conda_env, verbose=100) as exec:
        futs = [
            exec.submit(_run_tile, tilename, seed, opth, tmpdir)
            for tilename, seed in zip(tnames, seeds)
        ]
        for fut in PBar(as_completed(futs), total=len(futs), desc="running mdet"):
            pass
