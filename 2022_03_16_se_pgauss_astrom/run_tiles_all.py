import sys
import os
import fitsio
import numpy as np
import time
import subprocess
from functools import lru_cache
from contextlib import redirect_stdout, redirect_stderr
import random

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


# this function is from halotools
def crossmatch(x, y, skip_bounds_checking=False):
    r"""
    Finds where the elements of ``x`` appear in the array ``y``, including repeats.

    The elements in x may be repeated, but the elements in y must be unique.
    The arrays x and y may be only partially overlapping.

    The applications of this function envolve cross-matching
    two catalogs/data tables which share an objectID.
    For example, if you have a primary data table and a secondary data table containing
    supplementary information about (some of) the objects, the
    `~halotools.utils.crossmatch` function can be used to "value-add" the
    primary table with data from the second.

    For another example, suppose you have a single data table
    with an object ID column and also a column for a "host" ID column
    (e.g., ``halo_hostid`` in Halotools-provided catalogs),
    you can use the `~halotools.utils.crossmatch` function to create new columns
    storing properties of the associated host.

    See :ref:`crossmatching_halo_catalogs` and :ref:`crossmatching_galaxy_catalogs`
    for tutorials on common usages of this function with halo and galaxy catalogs.

    Parameters
    ----------
    x : integer array
        Array of integers with possibly repeated entries.

    y : integer array
        Array of unique integers.

    skip_bounds_checking : bool, optional
        The first step in the `crossmatch` function is to test that the input
        arrays satisfy the assumptions of the algorithm
        (namely that ``x`` and ``y`` store integers,
        and that all values in ``y`` are unique).
        If ``skip_bounds_checking`` is set to True,
        this testing is bypassed and the function evaluates faster.
        Default is False.

    Returns
    -------
    idx_x : integer array
        Integer array used to apply a mask to x
        such that x[idx_x] == y[idx_y]

    y_idx : integer array
        Integer array used to apply a mask to y
        such that x[idx_x] == y[idx_y]

    Notes
    -----
    The matching between ``x`` and ``y`` is done on the sorted arrays.  A consequence of
    this is that x[idx_x] and y[idx_y] will generally be a subset of ``x`` and ``y`` in
    sorted order.

    Examples
    --------
    Let's create some fake data to demonstrate basic usage of the function.
    First, let's suppose we have two tables of objects, ``table1`` and ``table2``.
    There are no repeated elements in any table, but these tables only
    partially overlap.
    The example below demonstrates how to transfer column data from ``table2``
    into ``table1`` for the subset of objects that appear in both tables.

    >>> num_table1 = int(1e6)
    >>> x = np.random.rand(num_table1)
    >>> objid = np.arange(num_table1)
    >>> from astropy.table import Table
    >>> table1 = Table({'x': x, 'objid': objid})

    >>> num_table2 = int(1e6)
    >>> objid = np.arange(5e5, num_table2+5e5)
    >>> y = np.random.rand(num_table2)
    >>> table2 = Table({'y': y, 'objid': objid})

    Note that ``table1`` and ``table2`` only partially overlap. In the code below,
    we will initialize a new ``y`` column for ``table1``, and for those rows
    with an ``objid`` that appears in both ``table1`` and ``table2``,
    we'll transfer the values of ``y`` from ``table2`` to ``table1``.

    >>> idx_table1, idx_table2 = crossmatch(table1['objid'].data, table2['objid'].data)
    >>> table1['y'] = np.zeros(len(table1), dtype = table2['y'].dtype)
    >>> table1['y'][idx_table1] = table2['y'][idx_table2]

    Now we'll consider a slightly more complicated example in which there
    are repeated entries in the input array ``x``. Suppose in this case that
    our data ``x`` comes with a natural grouping, for example into those
    galaxies that occupy a common halo. If we have a separate table ``y`` that
    stores attributes of the group, we may wish to broadcast some group property
    such as total group mass amongst all the group members.

    First create some new dummy data to demonstrate this application of
    the `crossmatch` function:

    >>> num_galaxies = int(1e6)
    >>> x = np.random.rand(num_galaxies)
    >>> objid = np.arange(num_galaxies)
    >>> num_groups = int(1e4)
    >>> groupid = np.random.randint(0, num_groups, num_galaxies)
    >>> galaxy_table = Table({'x': x, 'objid': objid, 'groupid': groupid})

    >>> groupmass = np.random.rand(num_groups)
    >>> groupid = np.arange(num_groups)
    >>> group_table = Table({'groupmass': groupmass, 'groupid': groupid})

    Now we use the `crossmatch` to paint the appropriate value of ``groupmass``
    onto each galaxy:

    >>> idx_galaxies, idx_groups = crossmatch(
        galaxy_table['groupid'].data, group_table['groupid'].data)
    >>> galaxy_table['groupmass'] = np.zeros(
        len(galaxy_table), dtype = group_table['groupmass'].dtype)
    >>> galaxy_table['groupmass'][idx_galaxies] = group_table['groupmass'][idx_groups]

    See the tutorials for additional demonstrations of alternative
    uses of the `crossmatch` function.

    See also
    ---------
    :ref:`crossmatching_halo_catalogs`

    :ref:`crossmatching_galaxy_catalogs`

    """
    # Ensure inputs are Numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Require that the inputs meet the assumptions of the algorithm
    if skip_bounds_checking is True:
        pass
    else:
        try:
            assert len(set(y)) == len(y)
            assert np.all(np.array(y, dtype=np.int64) == y)
            assert np.shape(y) == (len(y), )
        except Exception:
            msg = ("Input array y must be a 1d sequence of unique integers")
            raise ValueError(msg)
        try:
            assert np.all(np.array(x, dtype=np.int64) == x)
            assert np.shape(x) == (len(x), )
        except Exception:
            msg = ("Input array x must be a 1d sequence of integers")
            raise ValueError(msg)

    # Internally, we will work with sorted arrays, and then undo the sorting at the end
    idx_x_sorted = np.argsort(x)
    idx_y_sorted = np.argsort(y)
    x_sorted = np.copy(x[idx_x_sorted])
    y_sorted = np.copy(y[idx_y_sorted])

    # x may have repeated entries, so find the unique values
    # as well as their multiplicity
    unique_xvals, counts = np.unique(x_sorted, return_counts=True)

    # Determine which of the unique x values has a match in y
    unique_xval_has_match = np.in1d(unique_xvals, y_sorted, assume_unique=True)

    # Create a boolean array with True for each value in x with a match, otherwise False
    idx_x = np.repeat(unique_xval_has_match, counts)

    # For each unique value of x with a match in y, identify the index of the match
    matching_indices_in_y = np.searchsorted(
        y_sorted, unique_xvals[unique_xval_has_match]
    )

    # Repeat each matching index according to the multiplicity in x
    idx_y = np.repeat(matching_indices_in_y, counts[unique_xval_has_match])

    # Undo the original sorting and return the result
    return idx_x_sorted[idx_x], idx_y_sorted[idx_y]


def _download_file(fname, ntry=10):
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("./data/%s" % os.path.basename(fname)):

        for tt in range(ntry):
            try:

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
                break
            except Exception as e:
                if tt == ntry-1:
                    raise e
                else:
                    time.sleep(600 + random.randint(-100, 100))
                    pass

    return "./data/%s" % os.path.basename(fname)


def _get_piff_path_base(se_filename):
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


def _read_piff_base(fname):
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
    @lru_cache(maxsize=1024)
    def _read_piff(fn):
        return _read_piff_base(fn)

    @lru_cache(maxsize=1024)
    def _get_piff_path(fn):
        return _get_piff_path_base(fn)

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
