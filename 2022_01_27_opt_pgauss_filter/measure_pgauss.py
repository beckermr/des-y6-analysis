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
        dcat["flux_i"][rind]
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
    s2ns = []
    g1s = []
    flags = []
    for ap in aps:
        mom = PGaussMom(ap).go(obs)
        flags.append(mom["flags"])
        s2ns.append(mom["s2n"])
        g1s.append(mom["e1"])

    return s2ns, g1s, flags


def main():
    n_per_chunk = 100
    n_chunks = int(sys.argv[1])
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

    tilename = "DES0001-0624"
    mfile = _download_tile(tilename, ".")[2]
    m = meds.MEDS(mfile)

    wgt_cache = {}

    def _draw_noise(rng, m):
        rind = rng.randint(low=0, high=m.size-1)
        while m["ncutout"][rind] < 1:
            rind = rng.randint(low=0, high=m.size-1)
        if rind not in wgt_cache:
            wgt_cache[rind] = 1.0/np.sqrt(
                np.median(m.get_cutout(rind, 0, type="weight"))
            )

        return wgt_cache[rind], m.get_cutout(rind, 0, type="psf")

    aps = np.linspace(1.5, 2.5, 10)
    outputs = []
    for chunk in tqdm.trange(n_chunks):
        jobs = []
        for i in range(n_per_chunk):
            gal = _get_object(rng, dcat)
            nse, psf = _draw_noise(rng, m)
            jobs.append(joblib.delayed(_meas)(
                gal, psf, nse, aps, rng.randint(low=1, high=2**29))
            )

        with joblib.Parallel(n_jobs=-1, verbose=10) as par:
            outputs.extend(par(jobs))

        d = np.zeros(len(outputs), dtype=[
            ("s2n", "f4", (len(aps),)),
            ("e1", "f4", (len(aps),)),
            ("flags", "i4", (len(aps),))
        ])
        _o = np.array(outputs)
        d["s2n"] = _o[:, 0]
        d["e1"] = _o[:, 1]
        d["flags"] = _o[:, 2]

        fitsio.write("meas.fits", d, clobber=True)


if __name__ == "__main__":
    main()
