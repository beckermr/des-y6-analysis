#!/usr/bin/env bash

des-pizza-cutter \
  --config des-pizza-slices-y6-v9.yaml \
  --info=${MEDS_DIR}/des-pizza-slices-y6-v9/pizza_cutter_info/DES2132-5748_r_pizza_cutter_info.yaml \
  --output-path=`pwd` \
  --log-level=debug \
  --seed=3232 \
  --use-tmpdir \
  --n-jobs=4
