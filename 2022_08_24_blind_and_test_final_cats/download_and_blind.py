import os
import fitsio
import subprocess
import tqdm
import io
import sys
import contextlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from des_y6utils.shear_masking import generate_shear_masking_factor

OUTDIR = "blinded_data"
COLS_TO_KEEP = ["wmom", "pgauss", "pgauss_reg0.90"]


def _download(fname):
    cmd = """\
    rsync \
            -av \
            --password-file $DES_RSYNC_PASSFILE \
            ${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/%s \
            ./mdet_data/%s
    """ % (fname, os.path.basename(fname))
    subprocess.run(cmd, shell=True, check=True)


def _is_ok(fname):
    if os.path.exists(fname):
        try:
            fitsio.read_header(fname)
            return True
        except Exception:
            return False
    else:
        return False


def _msk_shear(fname, passphrase):
    if _is_ok(os.path.join(".", OUTDIR, os.path.basename(fname))):
        return None

    fac = generate_shear_masking_factor(passphrase)
    failed = False
    err = ""

    try:
        _download(fname)
        d = fitsio.read("./mdet_data/" + fname)
    except Exception as e:
        err = repr(e)
        failed = True

    if not failed:
        buff = io.StringIO()
        with contextlib.redirect_stderr(sys.stdout):
            with contextlib.redirect_stdout(buff):
                try:
                    d = fitsio.read(fname)

                    # make all non-zero flags nan in shear
                    for pre in COLS_TO_KEEP:
                        msk = (d[pre + "_flags"] != 0)
                        for col in [
                            '_g_1',
                            '_g_2',
                            '_g_cov_1_1',
                            '_g_cov_1_2',
                            '_g_cov_2_2',
                        ]:
                            d[pre + col][msk] = np.nan

                        # apply blinding to the good ones
                        msk = (
                            (d["mdet_step"] == "noshear")
                            & (d[pre + "_flags"] == 0)
                        )
                        e1o, e2o = (
                            d[pre + "_g_1"][msk].copy(),
                            d[pre + "_g_2"][msk].copy(),
                        )
                        e1 = e1o * fac
                        e2 = e2o * fac
                        d[pre + "_g_1"][msk] = e1
                        d[pre + "_g_2"][msk] = e2

                        assert not np.array_equal(d[pre + "_g_1"][msk], e1o)
                        assert not np.array_equal(d[pre + "_g_2"][msk], e2o)

                    msk = None
                    for pre in COLS_TO_KEEP:
                        _msk = (
                            (d[pre + "_flags"] == 0)
                            & (d[pre + "_s2n"] >= 5.0)
                        )
                        if msk is None:
                            msk = _msk
                        else:
                            msk |= _msk

                    msk &= (d["mask_flags"] == 0)

                    out = os.path.join(".", OUTDIR, os.path.basename(fname))
                    fitsio.write(out, d[msk], clobber=True)

                except Exception as e:
                    failed = True
                    err = repr(e)

    if failed:
        print("\ntile %s failed! - %s" % (fname, err), flush=True)
    else:
        pass
        # print("copied tile %s!" % fname, flush=True)


if __name__ == "__main__":
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists("./mdet_data"):
        os.makedirs("./mdet_data", exist_ok=True)

    with open(os.path.expanduser("~/.test_des_blinding_v7"), "r") as fp:
        passphrase = fp.read().strip()

    d = fitsio.read("fnames.fits", lower=True)

    fnames = sorted([
        os.path.join(d["path"][i], d["filename"][i])
        for i in range(len(d))
    ])[0:2]

    print("found %d tiles to process" % len(fnames), flush=True)
    with ProcessPoolExecutor(max_workers=10) as exec:
        futs = [
            exec.submit(_msk_shear, fname, passphrase)
            for fname in tqdm.tqdm(fnames, desc="making jobs")
        ]
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs)):
            try:
                fut.result()
            except Exception as e:
                print(e)

    os.system(f"cd {OUTDIR} && ls -1 *.fits > mdet_files.txt")
