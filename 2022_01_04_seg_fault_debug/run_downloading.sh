#!/usr/bin/env bash

for tname in DES0013-5457; do
  for band in z; do
    des-pizza-cutter-prep-tile \
      --config des-pizza-slices-y6-final.yaml \
      --tilename ${tname} \
      --band ${band}
  done
done
