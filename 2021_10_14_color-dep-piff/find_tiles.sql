select
    distinct
    i.tilename
from
    image i,
    image j,
    proctag tme,
    pfw_attempt_val av,
    proctag tse,
    file_archive_info fai
where
    tme.tag='Y6A2_COADD'
    and tme.pfw_attempt_id=av.pfw_attempt_id
    and av.pfw_attempt_id=i.pfw_attempt_id
    and i.filetype='coadd_nwgint'
    and i.band='z'
    and i.expnum=j.expnum
    and i.ccdnum=j.ccdnum
    and i.expnum = 226652
    and j.filetype='red_immask'
    and j.pfw_attempt_id=tse.pfw_attempt_id
    and tse.tag='Y6A1_COADD_INPUT'
    and fai.filename=j.filename;
