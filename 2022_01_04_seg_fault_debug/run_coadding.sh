#!/usr/bin/env bash

for band in z; do
  des-pizza-cutter \
    --config des-pizza-slices-y6-final.yaml \
    --info=${MEDS_DIR}/des-pizza-slices-y6-final/pizza_cutter_info/DES0013-5457_${band}_pizza_cutter_info.yaml \
    --output-path=`pwd` \
    --seed=3232 \
    --n-jobs=1 \
    --n-chunks=1 \
    --range=0:30 \
    --log-level=INFO
done
