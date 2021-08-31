#!/usr/bin/env bash

run-metadetect-on-slices \
  --config=metadetect-v3.yaml \
  --output-path=. \
  --seed=23487956 \
  --log-level=WARNING \
  --n-jobs=$1 \
  --range=0:10 \
  --use-tmpdir \
  DES0007-5957_r5227p01_z_pizza-cutter-slices.fits.fz
