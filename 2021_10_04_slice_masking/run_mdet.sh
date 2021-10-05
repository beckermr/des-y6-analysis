#!/usr/bin/env bash

tname="DES2229-3957"

run-metadetect-on-slices \
  --config=metadetect-v4.yaml \
  --output-path=. \
  --seed=23487956 \
  --log-level=WARNING \
  --n-jobs=1 \
  --range="0:20" \
  --viz-dir=./mbobs_plots \
  --band-names=r \
  ${tname}_r_des-pizza-slices-y6-v9_meds-pizza-slices-range0000-0020.fits.fz
