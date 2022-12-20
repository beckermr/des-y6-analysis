import sys
import fitsio
import numpy as np
import meds
from metadetect.fitting import fit_mbobs_list_wavg
import ngmix
from ngmix.medsreaders import NGMixMEDS, MultiBandNGMixMEDS
import more_itertools
import joblib


def _fit(jinds, mpths):
    m = None
    try:
        m = MultiBandNGMixMEDS([NGMixMEDS(pth) for pth in mpths])

        mbobs_list = [
            m.get_mbobs(i)
            for i in jinds
        ]
        fitter = ngmix.gaussmom.GaussMom(1.2)

        return fit_mbobs_list_wavg(
            mbobs_list=mbobs_list,
            fitter=fitter,
            bmask_flags=2**30,
            symmetrize=False,
        )
    finally:
        if m is not None:
            for _m in m.mlist:
                _m.close()


mfiles = sys.argv[1:]

m = meds.MEDS(mfiles[0])
n = m.size
m.close()

job_inds = list(more_itertools.chunked(range(n), n//99))

jobs = []
for jinds in job_inds:
    jobs.append(
        joblib.delayed(_fit)(jinds, mfiles)
    )

with joblib.Parallel(n_jobs=10, backend="multiprocessing", verbose=100) as exc:
    res = exc(jobs)

res = np.concatenate(res, axis=0)

fitsio.write("data.fits", res, clobber=True)
