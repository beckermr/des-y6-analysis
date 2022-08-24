#!/bin/bash

tag=ppsfperfv2

python -m cProfile -o prof_${tag}.dat `which run-metadetect-on-slices` \
  --config=metadetect-v6-all-meas.yaml \
  --output-path=./mdet_data \
  --seed=1342 \
  --use-tmpdir \
  --tmpdir=${TMPDIR} \
  --log-level=INFO \
  --n-jobs=1 \
  --band-names=griz \
  --range="0:20" \
  ./data/DES0328-2249_r5922p01_g_pizza-cutter-slices.fits.fz \
  ./data/DES0328-2249_r5922p01_r_pizza-cutter-slices.fits.fz \
  ./data/DES0328-2249_r5922p01_i_pizza-cutter-slices.fits.fz \
  ./data/DES0328-2249_r5922p01_z_pizza-cutter-slices.fits.fz

flameprof prof_${tag}.dat -o prof_${tag}.svg
