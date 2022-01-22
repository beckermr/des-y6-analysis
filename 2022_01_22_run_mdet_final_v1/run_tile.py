import sys
import os
import fitsio
import subprocess
import numpy as np
from concurrent.futures import as_completed
from esutil.pbar import PBar
from condor_exec import CondorExecutor


def _run_tile(tilename):
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
        subprocess.run(cmd, shell=True)
        mfiles.append("./data/%s" % os.path.basename(fname))


conda_env = "des-y6-final-v1"

tname = sys.argv[1]
_run_tile(tname)
