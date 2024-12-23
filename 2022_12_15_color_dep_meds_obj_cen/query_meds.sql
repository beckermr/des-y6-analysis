select
    concat(fai.filename, fai.compression) as filename,
    fai.path as path,
    m.tilename
from
    desfile d1,
    proctag t,
    miscfile m,
    file_archive_info fai
where
    d1.pfw_attempt_id = t.pfw_attempt_id
    and t.tag = 'Y6A2_MEDS_V3_TEST_V2'
    and d1.filename = m.filename
    and d1.id = fai.desfile_id
    and m.tilename = 'DES0253+0500'
    and fai.archive_name = 'desar2home'
    and d1.filetype = 'coadd_meds'; > fnames.fits
