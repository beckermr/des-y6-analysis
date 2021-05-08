import subprocess

with open("tiles.txt", "r") as fp:
    for line in fp.readlines():
        tname = line.strip()
        subprocess.run(
            "rsync -raP --password-file ${DES_RSYNC_PASSFILE} "
            "${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/ACT/multiepoch/Y6A2_PIZZACUTTER/r5227/%s "
            "${DESDATA}/ACT/multiepoch/Y6A2_PIZZACUTTER/r5227/" % tname,
            shell=True,
            check=True,
        )
