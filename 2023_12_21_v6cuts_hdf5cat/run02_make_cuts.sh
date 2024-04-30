#!/bin/bash

ls /gpfs02/astro/workarea/beckermr/des-y6-analysis/2023_02_25_run_mdet_nocoadd/mdet_data/*.fits > mdet_flist.txt
pizza-patches-make-cut-files \
    --flist=`pwd`/mdet_flist.txt \
    --uid-info=`pwd`/mdet_uids.yaml \
    --patches="/astro/u/esheldon/y6patches/patches-altrem-npatch200-seed8888.fits.gz" \
    --outdir=`pwd`/mdet_data_v6cuts \
    --keep-coarse-cuts

chmod go-rwx mdet_data_v6cuts/*.fits
chmod u-w mdet_data_v6cuts/*.fits
