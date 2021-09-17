#!/usr/bin/env bash

run-metadetect-on-slices \
  --config=metadetect-v4.yaml \
  --output-path=. \
  --seed=23487956 \
  --log-level=WARNING \
  --n-jobs=$1 \
  --range="0:$1" \
  DES2110-4748_z_des-pizza-slices-y6-v9_meds-pizza-slices.fits.fz
