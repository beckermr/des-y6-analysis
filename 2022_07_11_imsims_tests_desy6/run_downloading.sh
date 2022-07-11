#!/bin/bash

for tname in DES0003-3832; do
  for band in r i z; do
    des-pizza-cutter-prep-tile \
      --config des-pizza-slices-y6-v14.yaml \
      --tilename ${tname} \
      --band ${band}
  done
done
