#!/bin/bash

for tname in DES0433-2332; do
  for band in r; do
    des-pizza-cutter-prep-tile \
      --config des-pizza-slices-y6-v15-nostars.yaml \
      --tilename ${tname} \
      --band ${band}
  done
done
