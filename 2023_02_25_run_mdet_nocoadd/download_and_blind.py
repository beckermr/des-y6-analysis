import os
import fitsio
import subprocess
import tqdm
import io
import sys
import contextlib
import numpy as np
import glob
from ngmix.shape import g1g2_to_eta1eta2, eta1eta2_to_g1g2
from concurrent.futures import ProcessPoolExecutor, as_completed

from des_y6utils.shear_masking import generate_shear_masking_factor

OUTDIR = "blinded_data"
COLS_TO_KEEP = ["pgauss", "gauss"]


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
    indata = os.path.join(".", "mdet_data", os.path.basename(fname))
    outdata = os.path.join(".", OUTDIR, os.path.basename(fname))
    if _is_ok(outdata):
        return None

    fac = generate_shear_masking_factor(passphrase)
    failed = False
    err = ""

    if not failed:
        buff = io.StringIO()
        with contextlib.redirect_stderr(sys.stdout):
            with contextlib.redirect_stdout(buff):
                try:
                    d = fitsio.read(indata)

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
                        if pre not in ["gauss"]:
                            e1 = e1o * fac
                            e2 = e2o * fac
                        else:
                            # use eta due to bounds
                            eta1o, eta2o = g1g2_to_eta1eta2(e1o, e2o)
                            eta1 = eta1o * fac
                            eta2 = eta2o * fac
                            e1, e2 = eta1eta2_to_g1g2(eta1, eta2)

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
                    fitsio.write(outdata, d[msk], clobber=True)

                except Exception:
                    failed = True
                    err = "blinding error"

    if failed:
        print("\ntile %s failed! - %s" % (fname, err), flush=True)
    else:
        pass
        # print("copied tile %s!" % fname, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        num = int(sys.argv[1])
        tot = int(sys.argv[2])
    else:
        num = 0
        tot = 1

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists("./mdet_data"):
        os.makedirs("./mdet_data", exist_ok=True)
        os.system("chmod go-r ./mdet_data")

    with open(os.path.expanduser("~/.test_des_blinding_v7"), "r") as fp:
        passphrase = fp.read().strip()

    _fnames = glob.glob("./mdet_data/*.fits")
    _done_fnames = glob.glob("./%s/*.fits" % OUTDIR)
    _done_fnames = [
        os.path.basename(fname)
        for fname in _done_fnames
    ]

    fnames = [
        fname
        for i, fname in enumerate(_fnames)
        if i % tot == num and os.path.basename(fname) not in _done_fnames
    ]

    print(
        "found %d tiles to process for task %d" % (
            len(fnames), num
        ),
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=5) as exec:
        futs = [
            exec.submit(_msk_shear, fname, passphrase)
            for fname in tqdm.tqdm(fnames, desc="making jobs")
        ]
        for fut in tqdm.tqdm(
            as_completed(futs), total=len(futs), desc="making catalogs"
        ):
            try:
                fut.result()
            except Exception as e:
                print(e)

            if num == 0:
                os.system(f"cd {OUTDIR} && ls -1 *.fits > mdet_files.txt")
