import glob
import fitsio
import tqdm

for band in "griz":
    fns = glob.glob("epochs_info/%s/*.fits" % band)
    uccds = set()
    for fn in tqdm.tqdm(fns):
        ei = fitsio.read(fn)
        for i in range(len(ei)):
            uccds.add((ei["expnum"][i], ei["ccdnum"][i], ei["band"][i]))

    with open("uccds_%s.txt" % band, "w") as fp:
        fp.write("# expnum ccdnum band\n")
        for dta in uccds:
            fp.write("%d %d %s\n" % dta)
