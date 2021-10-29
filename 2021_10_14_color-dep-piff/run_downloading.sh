#!/usr/bin/env bash

for band in z; do
  des-pizza-cutter-prep-tile \
    --config des-pizza-slices-y6-test.yaml \
    --tilename DES2041-5248 \
    --band ${band}
done
