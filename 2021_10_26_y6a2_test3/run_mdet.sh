#!/usr/bin/env bash

run-metadetect-on-slices \
  --config=metadetect-v4.yaml \
  --output-path=. \
  --seed=23487956 \
  --log-level=WARNING \
  --n-jobs=2 \
  --range="0:20" \
  --band-names=griz \
  DES2120-4706_r5581p01_g_pizza-cutter-slices.fits.fz \
  DES2120-4706_r5581p01_r_pizza-cutter-slices.fits.fz \
  DES2120-4706_r5581p01_i_pizza-cutter-slices.fits.fz \
  DES2120-4706_r5581p01_z_pizza-cutter-slices.fits.fz
