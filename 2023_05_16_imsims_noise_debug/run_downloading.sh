#!/bin/bash

tname=$1
for band in r; do
  des-pizza-cutter-prep-tile \
    --config run001/des-pizza-slices-y6-v15-nostars.yaml \
    --tilename ${tname} \
    --band ${band}
done
