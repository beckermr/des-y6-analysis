#!/bin/bash

mkdir -p meds_dir
export MEDS_DIR=./meds_dir

band=$1
tname="DES0253+0500"

tmpmeds=/data/beckermr/2022_12_15_color_dep_meds_obj_cen/${tname}
mkdir -p ${tmpmeds}
rm -rf ${tmpmeds}/*

desmeds-make-meds-desdm \
  Y6A1_test_piff.yaml \
  ${MEDS_DIR}/Y6A1_test_piff/${tname}/lists-${band}/${tname}_${band}_fileconf-Y6A1_test_piff.yaml \
  --tmpdir=${tmpmeds} &> log${band}.oe

rm -rf ${tmpmeds}
