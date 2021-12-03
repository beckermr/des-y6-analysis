import os
import glob
from contextlib import redirect_stdout, redirect_stderr

import tqdm
from wurlitzer import pipes

fnames = glob.glob("**/*.fit*", recursive=True)
for fname in tqdm.tqdm(fnames, desc="removing fits files"):
    os.system("rm -f " + fname)

fnames = glob.glob("**/*.hs", recursive=True)
for fname in tqdm.tqdm(fnames, desc="removing healsparse files"):
    os.system("rm -f " + fname)

fnames = glob.glob("**/*.ipynb", recursive=True)
for fname in tqdm.tqdm(fnames, desc="clearing notebooks"):
    with pipes() as (out, err):
        with redirect_stdout(None):
            with redirect_stderr(None):
                os.system("jupyter nbconvert --clear-output --inplace " + fname)
