#!/usr/bin/env bash

tname="DES2229-3957"

run-metadetect-on-slices \
  --config=metadetect-v4.yaml \
  --output-path=. \
  --seed=23487956 \
  --log-level=WARNING \
  --n-jobs=8 \
  --range="0:200" \
  --band-names=griz \
  ${tname}_g_des-pizza-slices-y6-v9_meds-pizza-slices-range0000-0020.fits.fz \
  ${tname}_r_des-pizza-slices-y6-v9_meds-pizza-slices-range0000-0020.fits.fz \
  ${tname}_i_des-pizza-slices-y6-v9_meds-pizza-slices-range0000-0020.fits.fz \
  ${tname}_z_des-pizza-slices-y6-v9_meds-pizza-slices-range0000-0020.fits.fz
