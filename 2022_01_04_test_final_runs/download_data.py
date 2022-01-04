import os
import fitsio
import subprocess
import concurrent.futures


def _download(i):
    cmd = """\
    rsync \
            -av \
            --password-file $DES_RSYNC_PASSFILE \
            --files-from=fnames_%d.txt \
            ${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/ \
            ./data
    """ % i
    subprocess.run(cmd, shell=True)


n_threads = 4

d = fitsio.read("fnames.fits", lower=True)

fnames = [
    os.path.join(d["path"][i], d["filename"][i])
    for i in range(len(d))
]
nf_per = len(fnames)//n_threads

with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
    subprocess.run("mkdir -p data", shell=True)
    futs = []
    start = 0
    for i in range(n_threads):
        end = start + nf_per
        if end > len(fnames):
            end = len(fnames)

        with open("fnames_%d.txt" % i, "w") as fp:
            for fname in fnames[start:end]:
                fp.write("%s\n" % fname)

        start = end

        futs.append(executor.submit(_download, i))

    for fut in concurrent.futures.as_completed(futs):
        fut.result()
        print("done!")
