#!/usr/bin/env bash

run-metadetect-on-slices \
  --config=metadetect-v2.yaml \
  --output-path=. \
  --seed=23487956 \
  --log-level=WARNING \
  --range=0:210 \
  --n-jobs=$1 \
  DES2132-5748_r5227p01_i_pizza-cutter-slices.fits.fz
