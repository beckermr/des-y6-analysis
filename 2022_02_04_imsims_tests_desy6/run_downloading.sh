#!/bin/bash

for tname in DES2359-6539; do
  for band in r i z; do
    des-pizza-cutter-prep-tile \
      --config des-pizza-slices-y6-v12.yaml \
      --tilename ${tname} \
      --band ${band}
  done
done
