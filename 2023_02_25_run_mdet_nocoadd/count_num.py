import sys
import glob
import joblib
import fitsio


def _count_file(fname):
    return int(fitsio.read_header(fname, ext=1)["NAXIS2"])


if __name__ == "__main__":

    dr = sys.argv[1]
    fnames = glob.glob(f"{dr}/*.fits")
    jobs = [
        joblib.delayed(_count_file)(fname)
        for fname in fnames
    ]
    with joblib.Parallel(n_jobs=16, verbose=100, backend="loky") as exc:
        nums = exc(jobs)

    print(sum(nums))
