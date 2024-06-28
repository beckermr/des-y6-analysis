#!/usr/bin/env bash

nersc="9d6d994a-6d04-11e5-ba46-22000b92c6ec:/global/cfs/cdirs/des/y6-shear-catalogs/Y6A2_METADETECT_V6_UNBLINDED/tiles"
bnl="12782fb1-a599-4f18-b0fb-2e849681e214:/gpfs02/astro/workarea/beckermr/des-y6-analysis/2023_02_25_run_mdet_nocoadd/mdet_data"

for tname in `cat tiles.txt`; do
    fname=${tname}_metadetect-v10_mdetcat_part0000.fits
    echo ${fname}
    globus transfer ${bnl}/${fname} ${nersc}/${fname} --label ${tname}
done
