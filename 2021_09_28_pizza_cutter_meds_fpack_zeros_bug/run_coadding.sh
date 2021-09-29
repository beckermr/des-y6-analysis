#!/usr/bin/env bash

des-pizza-cutter \
  --config des-pizza-slices-y6-v9.yaml \
  --info=${MEDS_DIR}/des-pizza-slices-y6-v9/pizza_cutter_info/DES2229-3957_z_pizza_cutter_info.yaml \
  --output-path=`pwd` \
  --seed=3232 \
  --n-jobs=1 \
  --range="0:10" \
  --log-level=DEBUG
  # --use-tmpdir \
  # --tmpdir=/data/beckermr/tmp \
