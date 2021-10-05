#!/usr/bin/env bash

for band in g r i z; do
  des-pizza-cutter-prep-tile \
    --config des-pizza-slices-y6-v9.yaml \
    --tilename DES2229-3957 \
    --band ${band}
done
