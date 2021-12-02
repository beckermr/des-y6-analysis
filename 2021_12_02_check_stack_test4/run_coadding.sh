#!/usr/bin/env bash

for band in r; do
  des-pizza-cutter \
    --config des-pizza-slices-y6-test.yaml \
    --info=${MEDS_DIR}/des-pizza-slices-y6-test/pizza_cutter_info/DES0221-0750_${band}_pizza_cutter_info.yaml \
    --output-path=`pwd` \
    --seed=3232 \
    --log-level=DEBUG \
    --range="9005:9010"
    # --n-jobs=6 \
    # --n-chunks=12
  # --range="706:710" \
done
