#!/usr/bin/env bash

python -m pdb `which run-metadetect-on-slices` \
  --config=metadetect-v4.yaml \
  --output-path=. \
  --seed=23487956 \
  --log-level=WARNING \
  --n-jobs=$1 \
  --range="0:$1" \
  DES2132-5748_r_des-pizza-slices-y6-v9_meds-pizza-slices.fits.fz
