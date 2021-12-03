#!/usr/bin/env bash

for band in g r i z; do
  mkdir -p /data/beckermr/tmp/${band}
  des-pizza-cutter \
    --config des-pizza-slices-y6-test.yaml \
    --info=${MEDS_DIR}/des-pizza-slices-y6-test/pizza_cutter_info/DES0221-0750_${band}_pizza_cutter_info.yaml \
    --output-path=`pwd` \
    --seed=3232 \
    --n-jobs=1 \
    --n-chunks=1 \
    --tmpdir=/data/beckermr/tmp/${band} \
    --range=10:200
done
