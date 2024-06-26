import sys
import os
import fitsio
import subprocess
import numpy as np
import random
import time

from ngmix.medsreaders import NGMixMEDS
from esutil.pbar import PBar
from mattspy import BNLCondorParallel
import joblib


from ngmix.prepsfmom import PGaussMom


TESTING = False
WORKDIR = "/gpfs02/astro/workarea/beckermr/des-y6-analysis/2022_02_17_se_pgauss_cte"


def _query_gold(tilename, band):
    gf = WORKDIR + "/gold_ids/%s.fits" % tilename

    if not os.path.exists(gf):
        q = """\
    SELECT
        coadd_object_id
    FROM
        y6_gold_2_0
    WHERE
        flags_footprint > 0
        AND flags_gold = 0
        AND flags_foreground = 0
        AND ext_mash = 4
        AND tilename = '%s'; > gold_ids.fits
    """ % tilename
        with open("query.txt", "w") as fp:
            fp.write(q)
        subprocess.run("easyaccess --db dessci -l query.txt", shell=True, check=True)
        d = fitsio.read("gold_ids.fits")
        if band == "r":
            fitsio.write(gf, d, clobber=True)

        subprocess.run("rm -f gold_ids.fits", shell=True)
    else:
        d = fitsio.read(gf)

    return d["COADD_OBJECT_ID"]


def get_ccdnum(fname):
    return int(os.path.basename(fname).split("_")[2][1:])


def _download_tile(tilename, cwd, bands):
    os.system("mkdir -p data")

    d = fitsio.read(
        os.path.join(cwd, "meds_files.fits"),
        lower=True,
    )
    msk = (
        (d["tilename"] == tilename)
        & np.isin(d["band"], bands)
    )
    if np.sum(msk) != len(bands):
        return np.sum(msk)

    d = d[msk]
    mfiles = []
    for band in bands:
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


def _run_tile(tilename, band, seed, cwd):
    os.system("mkdir -p ./data")
    for _ in range(10):
        try:
            mfiles = _download_tile(tilename, cwd, [band])
            gids = _query_gold(tilename, band)
            break
        except Exception:
            time.sleep(600 + random.randint(-100, 100))
            mfiles = None
            gids = None
            pass

    if mfiles is None or gids is None:
        raise RuntimeError("Could not download files for tile %s" % tilename)
    elif not isinstance(mfiles, list):
        raise RuntimeError("Only found %d files for tile %s" % (mfiles, tilename))

    data = np.zeros((62, 32, 16), dtype=[
        ("e1", "f8"),
        ("e2", "f8"),
        ("row", "f8"),
        ("col", "f8"),
        ("n", "f8"),
        ("row_bin", "i4"),
        ("col_bin", "i4"),
        ("ccdnum", "i4"),
        ("band", "U1"),
        ("tilename", "U12"),
    ])
    for i in range(62):
        for j in range(32):
            for k in range(16):
                data["ccdnum"][i, j, k] = i+1
                data["row_bin"][i, j, k] = j
                data["col_bin"][i, j, k] = k
    data["band"] = band
    data["tilename"] = tilename

    fitter = PGaussMom(1.75)

    with NGMixMEDS(mfiles[0]) as m:
        ii = m.get_image_info()

        for i in PBar(range(m.size)):
            if m["id"][i] not in gids:
                continue
            for j in range(m["ncutout"][i]):
                if j > 0:
                    try:
                        o = m.get_obs(i, j, weight_type="uberseg")
                    except Exception:
                        o = None

                    if o is not None and np.all(o.weight > 0) and np.all(o.bmask == 0):
                        try:
                            res = fitter.go(o)
                        except Exception:
                            res = None

                        if (
                            res is not None
                            and res["flags"] == 0
                            and res["s2n"] > 10
                            and res["s2n"] < 100
                            and res["T"] > 0.5
                            and res["T"] < 1
                        ):
                            ccdnum = get_ccdnum(ii["image_path"][m["file_id"][i, j]])
                            if ccdnum >= 32:
                                rr = 4096 - m["orig_row"][i, j]
                            else:
                                rr = m["orig_row"][i, j]

                            cc = m["orig_col"][i, j]

                            rr_bin = int(max(min(rr-1, 4095), 0) / 128)
                            assert rr_bin >= 0, f"{rr_bin} is too low"
                            assert rr_bin < 32, f"{rr_bin} is too high"
                            cc_bin = int(max(min(cc-1, 2047), 0) / 128)
                            assert cc_bin >= 0, f"{cc_bin} is too low"
                            assert cc_bin < 16, f"{cc_bin} is too low"
                            data["e1"][ccdnum-1, rr_bin, cc_bin] += res["e"][0]
                            data["e2"][ccdnum-1, rr_bin, cc_bin] += res["e"][1]
                            data["row"][ccdnum-1, rr_bin, cc_bin] += rr
                            data["col"][ccdnum-1, rr_bin, cc_bin] += cc
                            data["n"][ccdnum-1, rr_bin, cc_bin] += 1
                            assert data["row_bin"][ccdnum-1, rr_bin, cc_bin] == rr_bin
                            assert data["col_bin"][ccdnum-1, rr_bin, cc_bin] == cc_bin

            if TESTING:
                break

    return data.ravel()


def main():
    cwd = os.path.abspath(os.path.realpath(os.getcwd()))
    seed = 100

    if len(sys.argv) == 1:
        d = fitsio.read(
            os.path.join(cwd, "meds_files.fits"),
            lower=True,
        )
        rng = np.random.RandomState(seed=seed)
        seeds = rng.randint(low=1, high=2**29, size=d.shape[0])
        tnames = d["tilename"]
    else:
        tnames = [sys.argv[1]]
        seeds = [10]

    if len(tnames) == 1:
        data = _run_tile(tnames[0], "i", seeds[0], cwd)
        fitsio.write(
            "%s_cte_data.fits" % tnames[0],
            data,
            clobber=True,
        )

    else:
        jobs = [
            joblib.delayed(_run_tile)(d["tilename"][i], d["band"][i], seeds[i], cwd)
            for i in range(d.shape[0])
            if d["band"][i] in ["r", "i", "z"]
        ]

        if TESTING:
            jobs = jobs[:101]

        all_data = []
        last_io = time.time()
        loc = 0
        io_loc = 0
        delta_io_loc = len(jobs) // 100
        with BNLCondorParallel(verbose=0, n_jobs=10000) as exc:
            for pr in PBar(exc(jobs), total=len(jobs), desc="running jobs"):
                try:
                    res = pr.result()
                except Exception as e:
                    print(f"\nfailure: {repr(e)}", flush=True)
                else:
                    all_data.append(res)
                    loc += 1

                if (
                    (loc % delta_io_loc == 0 or time.time() - last_io > 300)
                    and len(all_data) > 0
                ):
                    all_data = [np.concatenate(all_data, axis=0)]
                    fitsio.write(
                        "cte_data_all_%03d.fits" % io_loc,
                        all_data[0],
                        clobber=True,
                    )
                    last_io = time.time()

                    if loc % delta_io_loc == 0:
                        io_loc += 1
                        all_data = []

        if len(all_data) > 0:
            fitsio.write(
                "cte_data_all_%03d.fits" % io_loc,
                np.concatenate(all_data, axis=0),
                clobber=True,
            )


if __name__ == "__main__":
    main()
