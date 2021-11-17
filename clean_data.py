import os
import glob
import tqdm

fnames = glob.glob("**/*.fit*", recursive=True)

for fname in tqdm.tqdm(fnames):
    os.system("rm -f " + fname)
