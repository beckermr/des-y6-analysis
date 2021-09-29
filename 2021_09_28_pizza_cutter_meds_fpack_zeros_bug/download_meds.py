#!/usr/bin/python
import os
import sys
import subprocess

import easyaccess as ea


os.system("mkdir -p meds")

conn = ea.connect(section='desoper')
curs = conn.cursor()


tname = sys.argv[1]

query = """\
select
    fai.path,
    fai.filename as filename,
    fai.compression as compression
from proctag t, miscfile m, file_archive_info fai
where
    t.tag='Y6A2_COADD'
    and t.pfw_attempt_id=m.pfw_attempt_id
    and m.filetype='coadd_meds'
    and m.filename=fai.filename
    and m.tilename='%s'
""" % tname

curs.execute(query)

c = curs.fetchall()

for pth, fn, cz in c:
    cmd = """\
rsync \
    -raP \
    --password-file $DES_RSYNC_PASSFILE \
    ${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/%s meds/.""" % (os.path.join(pth, fn+cz))
    subprocess.run(cmd, shell=True)
