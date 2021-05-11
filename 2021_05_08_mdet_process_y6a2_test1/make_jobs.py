import os
import glob
import numpy as np

with open("tiles.txt", "r") as fp:
    tnames = fp.readlines()
    tnames = [t.strip() for t in tnames]

rng = np.random.RandomState(seed=7)
seeds = rng.randint(1, 2**30-1, size=len(tnames))

bands = ["r", "i", "z"]
os.makedirs("./jobs", exist_ok=True)
for seed, tname in zip(seeds, tnames):
    print("making job for tile %s" % tname, flush=True)

    # get the meds files in riz order
    all_mfiles = glob.glob(os.path.join(
        os.path.expandvars("${DESDATA}/ACT/multiepoch/Y6A2_PIZZACUTTER/r5227"),
        tname,
        "*/pizza-cutter/*.fits.fz",
    ))
    mfiles = []
    all_found = True
    for band in bands:
        found = False
        for fname in all_mfiles:
            if fname.endswith("_%s_pizza-cutter-slices.fits.fz" % band):
                found = True
                mfiles.append(fname)
                break
        all_found = all_found and found
    if not all_found:
        print("did not find all files for tile %s - skipping!" % tname, flush=True)
    else:
        for band, mfile in zip(bands, mfiles):
            print("    %s: %s" % (band, mfile), flush=True)

    # build wa script and job
    with open("./jobs/wq_%s.sub" % tname, "w") as fp:
        fp.write(f"""\
# one whole node
N: 1
mode: by_node
job_name: '{tname}'
command: |
  source ~/.bashrc
  conda activate des-y6a2-test1
  cd ~/workarea/des-y6-analysis/2021_05_08_mdet_process_y6a2_test1/outputs
  mkdir -p /data/beckermr-mdet-y6a2-test1/{tname}
  run-metadetect-on-slices \
    --config=../metadetect-v3.yaml \
    --output-path=. \
    --use-tmpdir \
    --tmpdir=/data/beckermr-mdet-y6a2-test1/{tname} \
    --seed={seed} \
    --log-level=WARNING \
    --n-jobs=12 \
    {mfiles[0]} {mfiles[1]} {mfiles[2]} || :
  rm -rf /data/beckermr-mdet-y6a2-test1/{tname}
""")
