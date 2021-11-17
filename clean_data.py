import os
import glob
import tqdm

fnames = glob.glob("**/*.fit*", recursive=True)

for fname in tqdm.tqdm(fnames, desc="removing fits files"):
    os.system("rm -f " + fname)

fnames = glob.glob("**/*.ipynb", recursive=True)
for fname in tqdm.tqdm(fnames, desc="clearing notebooks"):
    os.system("jupyter nbconvert --clear-output --inplace " + fname)
