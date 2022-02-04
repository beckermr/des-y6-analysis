import glob
import tqdm
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def _is_ok(fname):
    subprocess.run(
        "python -c 'import fitsio; fitsio.read(\"%s\")'" % fname,
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    fnames = sorted(glob.glob("mdet_data/*.fit*", recursive=True))
    print("found %d tiles to process" % len(fnames), flush=True)
    with ProcessPoolExecutor(max_workers=4) as exec:
        futs = {
            exec.submit(_is_ok, fname): fname
            for fname in tqdm.tqdm(fnames, desc="making jobs")
        }
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs)):
            try:
                fut.result()
            except Exception:
                print("\n" + futs[fut], flush=True)
