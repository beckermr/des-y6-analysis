import os
import sys
import subprocess

import meds
import ngmix
import numpy as np
import fitsio
import galsim
import joblib
import tqdm
from ngmix.prepsfmom import PGaussMom
from ngmix.admom import run_admom


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
                if not os.path.exists(os.path.join("./data", os.path.basename(fname))):
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


def _get_object(rng, dcat):
    rind = rng.randint(low=0, high=dcat.shape[0]-1)
    return (
        (
            (1.0 - dcat["bdf_fracdev"][rind])
            * galsim.Sersic(n=1, half_light_radius=dcat["bdf_hlr"][rind])
        )
        + (
            dcat["bdf_fracdev"][rind]
            * galsim.Sersic(n=4, half_light_radius=dcat["bdf_hlr"][rind])
        )
    ).shear(
        g1=dcat["bdf_g1"][rind], g2=dcat["bdf_g2"][rind]
    ).rotate(
        rng.uniform() * 360.0*galsim.degrees
    ).withFlux(
        dcat["flux_g"][rind]
        + dcat["flux_r"][rind]
        + dcat["flux_i"][rind]
        + dcat["flux_z"][rind]
    )


def _make_obs(gal, psf_im, nse, rng, n=49):
    psf = galsim.InterpolatedImage(galsim.ImageD(psf_im), scale=0.263)
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=n, ny=n, scale=0.263, method="no_pixel"
    ).array
    psf_im = psf.drawImage(nx=n, ny=n, scale=0.263, method="no_pixel").array
    cen = (n-1)/2

    im += rng.normal(size=im.shape, scale=nse)

    obs = ngmix.Observation(
        image=im,
        weight=np.ones_like(im)/nse**2,
        jacobian=ngmix.DiagonalJacobian(scale=0.263, row=cen, col=cen),
        psf=ngmix.Observation(
            image=psf_im,
            weight=np.ones_like(im),
            jacobian=ngmix.DiagonalJacobian(scale=0.263, row=cen, col=cen),
        ),
    )
    return obs


def _meas(gal, psf, nse, aps, seed):
    rng = np.random.RandomState(seed=seed)
    obs = _make_obs(
        gal,
        psf,
        nse,
        rng,
    )
    psf_mom = run_admom(obs.psf, 1.0, rng=rng)
    if psf_mom["flags"] == 0:
        psf_mom_t = psf_mom["T"]
    else:
        psf_mom_t = np.nan

    s2ns = []
    g1s = []
    ts = []
    trs = []
    flags = []
    for ap in aps:
        mom = PGaussMom(ap).go(obs)
        flags.append(mom["flags"] | psf_mom["flags"])
        s2ns.append(mom["s2n"])
        g1s.append(mom["e1"])
        ts.append(mom["T"])
        trs.append(mom["T"]/psf_mom_t)

    return s2ns, g1s, flags, ts, trs


def main():
    n_per_chunk = 1000
    n_chunks = 1
    n_tiles = int(sys.argv[1])
    seed = np.random.randint(low=1, high=2**29)
    rng = np.random.RandomState(seed=seed)

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    dcat = fitsio.read(os.path.expandvars("$MEDS_DIR/input_cosmos_v4.fits"))
    msk = (
        (dcat["mask_flags"] == 0)
        & (dcat["isgal"] == 1)
        & (dcat["bdf_hlr"] > 0)
        & (dcat["bdf_hlr"] < 5)
        & (dcat["mag_i"] <= 25)
    )
    dcat = dcat[msk]

    d = fitsio.read(
        "fnames.fits",
        lower=True,
    )
    tnames = sorted(list(set([
        d["filename"][i].split("_")[0]
        for i in range(d.shape[0])
    ])))

    aps = np.linspace(1.25, 2.5, 15)
    outputs = []
    with joblib.Parallel(n_jobs=-1, verbose=10, batch_size=2) as par:
        for _ in tqdm.trange(n_tiles):
            tilename = tnames[rng.randint(low=0, high=len(tnames)-1)]
            print("tile:", tilename, flush=True)
            mfiles = _download_tile(tilename, ".")
            ms = [meds.MEDS(mfile) for mfile in mfiles]

            wgt_cache = {}

            def _draw_noise(rng, ms):
                rind = rng.randint(low=0, high=ms[0].size-1)
                while any(m["ncutout"][rind] < 1 for m in ms):
                    rind = rng.randint(low=0, high=ms[0].size-1)
                if rind not in wgt_cache:
                    wgts = np.array([
                        1.0/np.median(m.get_cutout(rind, 0, type="weight"))
                        for m in ms
                    ])
                    wgt_cache[rind] = wgts

                psf = np.sum([
                    wgt * ms[i].get_cutout(rind, 0, type="psf")
                    for i, wgt in enumerate(wgt_cache[rind])
                ], axis=0)
                psf /= np.sum(psf)

                return np.sqrt(2/np.sum(1/wgt_cache[rind])), psf

            for chunk in tqdm.trange(n_chunks):
                jobs = []
                for i in range(n_per_chunk):
                    gal = _get_object(rng, dcat)
                    nse, psf = _draw_noise(rng, ms)
                    jobs.append(joblib.delayed(_meas)(
                        gal, psf, nse, aps, rng.randint(low=1, high=2**29))
                    )

                outputs.extend(par(jobs))

                d = np.zeros(len(outputs), dtype=[
                    ("s2n", "f4", (len(aps),)),
                    ("e1", "f4", (len(aps),)),
                    ("T", "f4", (len(aps),)),
                    ("Tratio", "f4", (len(aps),)),
                    ("flags", "i4", (len(aps),))
                ])
                _o = np.array(outputs)
                d["s2n"] = _o[:, 0]
                d["e1"] = _o[:, 1]
                d["flags"] = _o[:, 2]
                d["T"] = _o[:, 3]
                d["Tratio"] = _o[:, 4]

                fitsio.write(
                    "./results/meas_seed%d.fits" % seed,
                    d, extname="data", clobber=True)
                fitsio.write("./results/meas_seed%d.fits" % seed, aps, extname="aps")

            for m in ms:
                m.close()


if __name__ == "__main__":
    main()
