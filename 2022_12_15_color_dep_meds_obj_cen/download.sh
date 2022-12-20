#!/bin/bash

mkdir -p meds_dir
export MEDS_DIR=./meds_dir

tname="DES0253+0500"

for band in g r i z; do
  desmeds-prep-tile \
    --no-temp \
    --medsconf=Y6A1_test_piff.yaml \
    --tilename=${tname} \
    --band=${band}
done