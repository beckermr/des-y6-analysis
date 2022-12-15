#!/bin/bash

mkdir -p meds_dir
export MEDS_DIR=./meds_dir

tmpmeds=/data/beckermr/2022_12_15_color_dep_meds_obj_cen/DES0137-3749
mkdir -p ${tmpmeds}
rm -rf ${tmpmeds}/*

for band in g r i z; do
  desmeds-make-meds-desdm \
    Y6A1_test_piff.yaml \
    ${MEDS_DIR}/Y6A1_test_piff/DES0137-3749/lists-${band}/DES0137-3749_${band}_fileconf-Y6A1_test_piff.yaml \
    --tmpdir=${tmpmeds} &> log${band}.oe
done

rm -rf ${tmpmeds}