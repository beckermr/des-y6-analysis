#!/usr/bin/env bash

for tname in DES0131-3206; do
  for band in r; do
    des-pizza-cutter-prep-tile \
      --config test-y6-sims.yaml \
      --tilename ${tname} \
      --band ${band}
  done
done
