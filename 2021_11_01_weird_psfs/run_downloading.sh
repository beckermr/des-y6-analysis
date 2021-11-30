#!/usr/bin/env bash

for band in r; do
  des-pizza-cutter-prep-tile \
    --config des-pizza-slices-y6-test.yaml \
    --tilename DES0221-0750 \
    --band ${band}
done

# for tname in DES0131-3206 DES0137-3749 DES0221-0750 DES0229-0416; do
#   for band in g r i z; do
#     des-pizza-cutter-prep-tile \
#       --config des-pizza-slices-y6-test.yaml \
#       --tilename ${tname} \
#       --band ${band}
#   done
# done
