#!/usr/bin/env bash

for tname in DES0221-0750; do
  for band in g r i z; do
    des-pizza-cutter-prep-tile \
      --config des-pizza-slices-y6-test.yaml \
      --tilename ${tname} \
      --band ${band}
  done
done
