#!/bin/bash

mkdir -p meds_dir
export MEDS_DIR=./meds_dir

desmeds-prep-tile \
  --no-temp \
  --medsconf=Y6A1_test_piff.yaml \
  --tilename=DES0137-3749 \
  --band="r"
