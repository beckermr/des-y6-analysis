import os
import glob
import fitsio
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def _is_ok(fname):
    if os.path.exists(fname):
        try:
            fitsio.read(fname)
            return True
        except Exception:
            return False
    else:
        return False


if __name__ == "__main__":
    fnames = sorted(glob.glob("mdet_data/*.fit*", recursive=True))
    print("found %d tiles to process" % len(fnames), flush=True)
    with ProcessPoolExecutor(max_workers=8) as exec:
        futs = {
            exec.submit(_is_ok, fname): fname
            for fname in tqdm.tqdm(fnames, desc="making jobs")
        }
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs)):
            try:
                fut.result()
            except Exception:
                print("\n\n\n" + futs[fut] + "\n\n\n", flush=True)
