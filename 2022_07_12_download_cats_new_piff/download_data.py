import os
import fitsio
import subprocess
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _download(fname):
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     fpth = os.path.join(tmpdir, "fname_%s.txt" % os.path.basename(fname))
    #     with open(fpth) as fp:
    #         fp.write(fname + "\n")

    cmd = """\
    rsync \
            -av \
            --password-file $DES_RSYNC_PASSFILE \
            ${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/%s \
            ./mdet_data/%s
    """ % (fname, fname)
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    d = fitsio.read("fnames.fits", lower=True)

    fnames = [
        os.path.join(d["path"][i], d["filename"][i])
        for i in range(len(d))
    ]

    subprocess.run("mkdir -p mdet_data", shell=True)

    print("found %d tiles to download" % len(fnames), flush=True)
    with ThreadPoolExecutor(max_workers=10) as exec:
        futs = [
            exec.submit(_download, fname)
            for fname in tqdm.tqdm(fnames, desc="making jobs")
        ]
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs)):
            try:
                fut.result()
            except Exception as e:
                print(e)
