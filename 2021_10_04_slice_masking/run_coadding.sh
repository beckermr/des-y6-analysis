#!/usr/bin/env bash

des-pizza-cutter \
  --config des-pizza-slices-y6-v9.yaml \
  --info=${MEDS_DIR}/des-pizza-slices-y6-v9/pizza_cutter_info/DES2229-3957_r_pizza_cutter_info.yaml \
  --output-path=`pwd` \
  --seed=3232 \
  --n-jobs=1 \
  --log-level=DEBUG \
  --range="706:710"
  # --range="706:710" \
  # --use-tmpdir \
  # --tmpdir=/data/beckermr/tmp \
