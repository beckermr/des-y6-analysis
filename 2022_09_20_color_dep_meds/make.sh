#!/bin/bash

mkdir -p meds_dir
export MEDS_DIR=./meds_dir

tmpmeds=/data/beckermr/2022_09_20_color_dep_meds/DES0137-3749
mkdir -p ${tmpmeds}

desmeds-make-meds-desdm \
  Y6A1_test_piff.yaml \
  ${MEDS_DIR}/Y6A1_test_piff/DES0137-3749/lists-r/DES0137-3749_r_fileconf-Y6A1_test_piff.yaml \
  --tmpdir=${tmpmeds}

rm -rf ${tmpmeds}