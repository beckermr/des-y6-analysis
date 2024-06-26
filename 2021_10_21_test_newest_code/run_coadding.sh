#!/usr/bin/env bash

for band in g r i z; do
  mkdir -p /data/beckermr/tmp/${band}
  des-pizza-cutter \
    --config des-pizza-slices-y6-test.yaml \
    --info=${MEDS_DIR}/des-pizza-slices-y6-test/pizza_cutter_info/DES2041-5248_${band}_pizza_cutter_info.yaml \
    --output-path=`pwd` \
    --seed=3232 \
    --use-tmpdir \
    --tmpdir=/data/beckermr/tmp/${band} \
    --range="0:200"
    # --n-jobs=6 \
    # --n-chunks=12
  # --range="706:710" \
done
