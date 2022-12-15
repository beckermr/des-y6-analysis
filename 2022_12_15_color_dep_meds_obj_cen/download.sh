#!/bin/bash

mkdir -p meds_dir
export MEDS_DIR=./meds_dir

for band in g r i z; do
  desmeds-prep-tile \
    --no-temp \
    --medsconf=Y6A1_test_piff.yaml \
    --tilename=DES2214-5914 \
    --band=${band}
done