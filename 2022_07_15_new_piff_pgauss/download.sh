#!/bin/bash

for fname in \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0322-2249/p04/pizza-cutter/DES0322-2249_r5922p04_g_pizza-cutter-slices.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0322-2249/p04/pizza-cutter/DES0322-2249_r5922p04_i_pizza-cutter-slices.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0322-2249/p04/pizza-cutter/DES0322-2249_r5922p04_metadetect.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0322-2249/p04/pizza-cutter/DES0322-2249_r5922p04_r_pizza-cutter-slices.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0322-2249/p04/pizza-cutter/DES0322-2249_r5922p04_z_pizza-cutter-slices.fits.fz \
\
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0325-2249/p02/pizza-cutter/DES0325-2249_r5922p02_g_pizza-cutter-slices.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0325-2249/p02/pizza-cutter/DES0325-2249_r5922p02_i_pizza-cutter-slices.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0325-2249/p02/pizza-cutter/DES0325-2249_r5922p02_metadetect.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0325-2249/p02/pizza-cutter/DES0325-2249_r5922p02_r_pizza-cutter-slices.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0325-2249/p02/pizza-cutter/DES0325-2249_r5922p02_z_pizza-cutter-slices.fits.fz \
\
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0328-2249/p02/pizza-cutter/DES0328-2249_r5922p02_g_pizza-cutter-slices.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0328-2249/p02/pizza-cutter/DES0328-2249_r5922p02_i_pizza-cutter-slices.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0328-2249/p02/pizza-cutter/DES0328-2249_r5922p02_metadetect.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0328-2249/p02/pizza-cutter/DES0328-2249_r5922p02_r_pizza-cutter-slices.fits.fz \
  OPS/multiepoch/Y6A2_PIZZACUTTER/r5922/DES0328-2249/p02/pizza-cutter/DES0328-2249_r5922p02_z_pizza-cutter-slices.fits.fz \
\
; do

  rsync \
    -avP \
    --password-file $DES_RSYNC_PASSFILE \
    ${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/${fname} `basename ${fname}`
done
