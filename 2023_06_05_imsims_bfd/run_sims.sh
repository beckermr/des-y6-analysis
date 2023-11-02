#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}
export TMPDIR=/data/beckermr/tmp

# 1 - run number
# 2 - tilename
# 3 - seed
# 4 - gal mag

pout=./sim_outputs

mkdir -p ${pout}
run-eastlake-sim \
  -v 1 \
  --seed 42 \
  eastlake-config.yaml \
  ${pout} stamp.shear.g1=0.02 stamp.shear.g2=0.0 gal.flux.fgmag=18 output.tilename="DES0518-6039" $@
