#!/usr/bin/env bash

for band in r; do
  mkdir -p /data/beckermr/tmp/${band}
  des-pizza-cutter \
    --config des-pizza-slices-y6-v9-small.yaml \
    --info=${MEDS_DIR}/des-pizza-slices-y6-v9/pizza_cutter_info/DES2229-3957_${band}_pizza_cutter_info.yaml \
    --output-path=`pwd` \
    --seed=3232 \
    --use-tmpdir \
    --tmpdir=/data/beckermr/tmp/${band} \
    --n-jobs=6
done
  # --range="706:710" \
