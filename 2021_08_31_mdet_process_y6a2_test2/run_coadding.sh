#!/usr/bin/env bash

des-pizza-cutter \
  --config des-pizza-slices-y6-v9.yaml \
  --info=${MEDS_DIR}/des-pizza-slices-y6-v9/pizza_cutter_info/DES0156-3415_z_pizza_cutter_info.yaml \
  --output-path=`pwd` \
  --seed=3232 \
  --use-tmpdir \
  --tmpdir=/data/beckermr/tmp \
  --n-jobs=8
