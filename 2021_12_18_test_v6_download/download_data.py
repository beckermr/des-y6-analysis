import os
import fitsio
import subprocess


d = fitsio.read("fnames.fits", lower=True)

fnames = [
    os.path.join(d["path"][i], d["filename"][i])
    for i in range(len(d))
]

with open("fnames.txt", "w") as fp:
    for fname in fnames:
        fp.write("%s\n" % fname)

subprocess.run("mkdir -p data", shell=True)

cmd = """\
rsync \
        -av \
        --password-file $DES_RSYNC_PASSFILE \
        --files-from=fnames.txt \
        ${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/ \
        ./data
"""
subprocess.run(cmd, shell=True)
