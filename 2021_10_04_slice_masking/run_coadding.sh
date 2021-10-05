#!/usr/bin/env bash

for band in r i z; do
  des-pizza-cutter \
    --config des-pizza-slices-y6-v9.yaml \
    --info=${MEDS_DIR}/des-pizza-slices-y6-v9/pizza_cutter_info/DES2229-3957_${band}_pizza_cutter_info.yaml \
    --output-path=`pwd` \
    --seed=3232 \
    --use-tmpdir \
    --tmpdir=/data/beckermr/tmp \
    --n-jobs=2 &
done
  # --range="706:710" \
