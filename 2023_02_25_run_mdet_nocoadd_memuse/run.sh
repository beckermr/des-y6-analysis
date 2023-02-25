#!/bin/bash

python -m memray run -o output.bin `which run-metadetect-on-slices` \
  --config=./metadetect-v10.yaml \
  --output-path=./mdet_data \
  --seed=100 \
  --log-level=INFO \
  --n-jobs=2 \
  --range=0:10 \
  --band-names=griz \
  ./data/DES0300-2915_r5922p01_g_pizza-cutter-slices.fits.fz \
  ./data/DES0300-2915_r5922p01_r_pizza-cutter-slices.fits.fz \
  ./data/DES0300-2915_r5922p01_i_pizza-cutter-slices.fits.fz \
  ./data/DES0300-2915_r5922p01_z_pizza-cutter-slices.fits.fz
