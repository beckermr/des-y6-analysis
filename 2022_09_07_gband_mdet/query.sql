select
    concat(fai.filename, fai.compression) as filename,
    fai.path as path
from
    desfile d1,
    proctag t,
    miscfile m,
    file_archive_info fai
where
    d1.pfw_attempt_id = t.pfw_attempt_id
    and t.tag = 'Y6A2_METADETECT_V3'
    and d1.filename = m.filename
    and d1.id = fai.desfile_id
    and fai.archive_name = 'desar2home'
    and m.band = 'g'
    and d1.filetype = 'coadd_pizza_cutter'; > fnames.fits
