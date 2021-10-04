#!/usr/bin/env bash

tname="DES2229-3957"

run-metadetect-on-slices \
  --config=metadetect-v4.yaml \
  --output-path=. \
  --seed=23487956 \
  --log-level=WARNING \
  --n-jobs=1 \
  --range="$1:$2" \
  --viz-dir=./mbobs_plots \
  --band-names=riz \
  ${tname}_r5366p01_r_pizza-cutter-slices.fits.fz \
  ${tname}_r5366p01_i_pizza-cutter-slices.fits.fz \
  ${tname}_r5366p01_z_pizza-cutter-slices.fits.fz
