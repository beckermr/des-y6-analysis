import os
import glob

with open("tiles.txt", "r") as fp:
    tnames = fp.readlines()
    tnames = [t.strip() for t in tnames]

bands = ["r", "i", "z"]
os.makedirs("./jobs", exist_ok=True)
for tname in tnames:
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

    # build wa script and job
    
