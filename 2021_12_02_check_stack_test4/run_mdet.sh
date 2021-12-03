#!/usr/bin/env bash

tname="DES0221-0750"
ttail="des-pizza-slices-y6-test_meds-pizza-slices-range0010-0200.fits.fz"

run-metadetect-on-slices \
  --config=metadetect-v5.yaml \
  --output-path=. \
  --seed=23487956 \
  --log-level=WARNING \
  --n-jobs=8 \
  --range="10:200" \
  --band-names=griz \
  ${tname}_g_${ttail} \
  ${tname}_r_${ttail} \
  ${tname}_i_${ttail} \
  ${tname}_z_${ttail}
