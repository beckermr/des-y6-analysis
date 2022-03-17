import sys
import os
import fitsio
import numpy as np
import time
import subprocess
from functools import lru_cache
from contextlib import redirect_stdout, redirect_stderr

from crossmatch_utils import crossmatch
import piff
from ngmix.prepsfmom import PGaussMom
from ngmix.medsreaders import NGMixMEDS
import galsim
import wurlitzer
from esutil.pbar import PBar
from mattspy import BNLCondorParallel
import joblib
import tqdm


WORKDIR = "/gpfs02/astro/workarea/beckermr/des-y6-analysis/2022_03_16_se_pgauss_astrom"
if not os.path.exists(WORKDIR):
    WORKDIR = "."


def _download_file(fname):
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("./data/%s" % os.path.basename(fname)):
        cmd = """\
rsync \
-avP \
--password-file $DES_RSYNC_PASSFILE \
${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/%s \
./data/%s
    """ % (fname, os.path.basename(fname))
        s = subprocess.run(cmd, shell=True, capture_output=True)
        if s.returncode != 0:
            print(
                "download failed: %s" % (fname),
                flush=True,
            )
            raise RuntimeError(
                "download failed: %s %s" % (
                    s.stdout.decode("utf-8"), s.stderr.decode("utf-8"))
            )

    return "./data/%s" % os.path.basename(fname)


@lru_cache(maxsize=1024)
def _get_piff_path(se_filename):
    os.makedirs(WORKDIR + "/piff_paths", exist_ok=True)
    gf = WORKDIR + "/piff_paths/%s.txt" % se_filename
    if not os.path.exists(gf):
        parts = se_filename.split("_")
        expnum = int(parts[0][1:])
        ccdnum = int(parts[2][1:])

        _query = """
        select
            d2.filename as redfile,
            fai.filename as filename,
            fai.path as path,
            m.band as band,
            m.expnum as expnum,
            m.ccdnum as ccdnum
        from
            desfile d1,
            desfile d2,
            proctag t,
            opm_was_derived_from wdf,
            miscfile m,
            file_archive_info fai
        where
            d2.filename = '%s'
            and d2.id = wdf.parent_desfile_id
            and wdf.child_desfile_id = d1.id
            and d1.filetype = 'piff_model'
            and d1.pfw_attempt_id = t.pfw_attempt_id
            and t.tag = 'Y6A2_PIFF_V2'
            and d1.filename = m.filename
            and d1.id = fai.desfile_id
            and fai.archive_name = 'desar2home'
        """ % (se_filename[:-3] if se_filename.endswith(".fz") else se_filename)

        piff_file = None

        with wurlitzer.pipes():
            with redirect_stderr(None), redirect_stdout(None):
                import easyaccess as ea
                conn = ea.connect(section='desoper')
                curs = conn.cursor()
                curs.execute(_query)
        for row in curs:
            if row[4] == expnum and row[5] == ccdnum:
                piff_file = os.path.join(row[2], row[1])
        if piff_file is None:
            raise RuntimeError("could not find piff model for %s" % se_filename)

        with open(gf, "w") as fp:
            fp.write(piff_file)

    else:
        with open(gf, "r") as fp:
            piff_file = fp.read().strip()

    return piff_file


@lru_cache(maxsize=200)
def _read_piff(fname):
    return piff.read(fname)


def _query_gold(tilename, band):
    os.makedirs(WORKDIR + "/gold_ids", exist_ok=True)
    gf = WORKDIR + "/gold_ids/%s.fits" % tilename

    if not os.path.exists(gf):
        q = """\
    SELECT
        coadd_object_id,
        mag_auto_g - mag_auto_i as mag_auto_gmi
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
        s = subprocess.run(
            "easyaccess --db dessci -l query.txt", shell=True,
            capture_output=True,
        )
        if s.returncode != 0:
            print(
                "gold ids failed: %s" % (tilename),
                flush=True,
            )
            raise RuntimeError(
                "gold ids failed: %s %s %s" % (
                    tilename, s.stdout.decode("utf-8"), s.stderr.decode("utf-8"))
            )

        d = fitsio.read("gold_ids.fits")
        if band == "r":
            fitsio.write(gf, d, clobber=True)

        subprocess.run("rm -f gold_ids.fits", shell=True)
    else:
        d = fitsio.read(gf)

    return d["COADD_OBJECT_ID"], d["MAG_AUTO_GMI"]


def _draw_piff(x, y, pmod, color, use_piff_rend=False):
    PIFF_STAMP_SIZE = 25
    wcs = list(pmod.wcs.values())[0]
    chipnum = list(pmod.wcs.keys())[0]

    # compute the lower left corner of the stamp
    # we find the nearest pixel to the input (x, y)
    # and offset by half the stamp size in pixels
    # assumes the stamp size is odd
    # there is an assert for this below
    half = (PIFF_STAMP_SIZE - 1) / 2
    x_cen = np.floor(x+0.5)
    y_cen = np.floor(y+0.5)

    # make sure this is true so pixel index math is ok
    assert y_cen - half == int(y_cen - half)
    assert x_cen - half == int(x_cen - half)

    # compute bounds in Piff wcs coords
    xmin = int(x_cen - half)
    ymin = int(y_cen - half)

    bounds = galsim.BoundsI(
        xmin, xmin+PIFF_STAMP_SIZE-1,
        ymin, ymin+PIFF_STAMP_SIZE-1,
    )
    image = galsim.ImageD(bounds, wcs=wcs)
    if use_piff_rend:
        return pmod.draw(
            x=x,
            y=y,
            chipnum=chipnum,
            GI_COLOR=color,
            stamp_size=25,
        )

    else:
        return pmod.draw(
            x=x,
            y=y,
            chipnum=chipnum,
            image=image,
            GI_COLOR=color,
        )


def _run_tile(tilename, band, seed, cwd, desdm_path, base_color):
    gold_ids, gmi_gold = _query_gold(tilename, band)
    meds_pth = _download_file(desdm_path)
    with NGMixMEDS(meds_pth) as mfile:
        ii = mfile.get_image_info()
        ind_meds, ind_gold = crossmatch(mfile["id"], gold_ids)

        de1arr = []
        de2arr = []
        for i, gmi, gid in tqdm.tqdm(
            zip(ind_meds, gmi_gold[ind_gold], gold_ids[ind_gold]),
            ncols=79,
            total=len(ind_meds),
        ):
            assert mfile["id"][i] == gid

            ncutout = mfile["ncutout"][i]
            if ncutout <= 1:
                continue

            for k in range(1, ncutout):
                try:
                    obs = mfile.get_obs(i, k)
                except Exception:
                    continue

                if np.any(obs.weight == 0) or np.any((obs.bmask & 2**30) != 0):
                    continue

                res = PGaussMom(2).go(obs)
                pres = PGaussMom(2).go(obs.psf, no_psf=True)
                if (
                    res["flags"] == 0
                    and res["s2n"] > 10
                    and pres["flags"] == 0
                    and res["T"]/pres["T"] > 0.5
                ):
                    fname = os.path.basename(ii["image_path"][k])[:-3]
                    piff_file = _get_piff_path(fname)
                    try:
                        piff_file = _download_file(piff_file)
                        pmod = _read_piff(piff_file)
                    except Exception:
                        os.system("rm -f %s" % piff_file)
                        print(
                            "ERROR: piff file %s could not be found" % piff_file,
                            flush=True,
                        )
                    wcs = list(pmod.wcs.values())[0]
                    xy1 = np.array(wcs.radecToxy(
                        mfile["ra"][i], mfile["dec"][i], "degrees", color=base_color))
                    xy2 = np.array(wcs.radecToxy(
                        mfile["ra"][i], mfile["dec"][i], "degrees", color=gmi))
                    xy = (xy1 + xy2)/2
                    dxy = xy1-xy2
                    jac = wcs.jacobian(
                        image_pos=galsim.PositionD(xy[0], xy[1]), color=base_color)
                    du = jac.dudx * dxy[0] + jac.dudy * dxy[1]
                    dv = jac.dvdx * dxy[0] + jac.dvdy * dxy[1]
                    dxy = np.array([du, dv])
                    de1arr.append((dxy[0]**2 - dxy[1]**2)/res["T"])
                    de2arr.append(2*dxy[0]*dxy[1]/res["T"])

                    if len(de1arr) % 100 == 0 and len(de1arr) > 2:
                        print("\nmeds object %d" % i, flush=True)
                        print("    e1 [10^-4, 3sigma]: %0.4f +/- %0.4f" % (
                            np.mean(de1arr)/1e-4,
                            3*np.std(de1arr)/np.sqrt(len(de1arr))/1e-4
                        ), flush=True)
                        print("    e2 [10^-4, 3sigma]: %0.4f +/- %0.4f" % (
                            np.mean(de2arr)/1e-4,
                            3*np.std(de2arr)/np.sqrt(len(de2arr))/1e-4
                        ), flush=True)

    sz = len(de1arr)
    if sz > 0:
        d = np.zeros(
            sz,
            dtype=[("de1", "f8"), ("de2", "f8"), ("band", "U1"), ("tilename", "U12")]
        )
        d["de1"] = np.array(de1arr)
        d["de2"] = np.array(de2arr)
        d["band"] = band
        d["tilename"] = tilename

        return d
    else:
        return None


def main():
    cwd = os.path.abspath(os.path.realpath(os.getcwd()))
    seed = 100
    base_color = 1.4
    rng = np.random.RandomState(seed=seed)

    d = fitsio.read(
        os.path.join(cwd, "meds_files.fits"),
        lower=True,
    )

    if len(sys.argv) > 1:
        inds = rng.choice(d.shape[0], replace=False, size=int(sys.argv[1]))
        d = d[inds]

    if d.shape[0] == 1:
        use_joblib = True
    else:
        use_joblib = False

    seeds = rng.randint(low=1, high=2**29, size=d.shape[0])

    jobs = [
        joblib.delayed(_run_tile)(
            d["tilename"][i],
            d["band"][i],
            seeds[i],
            cwd,
            os.path.join(d["path"][i], d["filename"][i]),
            base_color,
        )
        for i in range(d.shape[0])
        if d["band"][i] in ["g", "r", "i"]
    ]

    if use_joblib:
        with joblib.Parallel(n_jobs=1, backend="sequential", verbose=0) as par:
            res = par(jobs)
        all_data = [r for r in res if r is not None]
        io_loc = 0
    else:
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
                    if res is not None:
                        all_data.append(res)
                        loc += 1

                if (
                    (loc % delta_io_loc == 0 or time.time() - last_io > 300)
                    and len(all_data) > 0
                ):
                    all_data = [np.concatenate(all_data, axis=0)]
                    fitsio.write(
                        "astrom_data_all_basecolor%0.2f_%03d.fits" % (
                            base_color, io_loc
                        ),
                        all_data[0],
                        clobber=True,
                    )
                    last_io = time.time()

                    if loc % delta_io_loc == 0:
                        io_loc += 1
                        all_data = []

    if len(all_data) > 0:
        fitsio.write(
            "astrom_data_all_basecolor%0.2f_%03d.fits" % (
                base_color, io_loc
            ),
            np.concatenate(all_data, axis=0),
            clobber=True,
        )


if __name__ == "__main__":
    main()
