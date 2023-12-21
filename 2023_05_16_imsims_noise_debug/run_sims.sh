#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}
export TMPDIR=/data/beckermr/tmp

# 1 - run number
# 2 - tilename
# 3 - seed
# 4 - gal mag

pout=./sim_outputs_plus_$2_$3
mout=./sim_outputs_minus_$2_$3

mkdir -p ${pout}
run-eastlake-sim \
  -v 1 \
  --seed $3 \
  eastlake-config.yaml \
  ${pout} stamp.shear.g1=0.02 stamp.shear.g2=0.0 gal.flux.fgmag=$4 output.tilename=$2

mkdir -p ${mout}
run-eastlake-sim \
  -v 1 \
  --seed $3 \
  eastlake-config.yaml \
  ${mout} stamp.shear.g1=-0.02 stamp.shear.g2=0.0 gal.flux.fgmag=$4 output.tilename=$2

  # --step_names pizza_cutter metadetect \
  # --resume_from=./sim_outputs/job_record.pkl \

  # --skip_completed_steps \
  # --resume_from=./sim_outputs/job_record.pkl \

#  --step_names src_extractor \
#  --resume_from=./sim_outputs/job_record.pkl \
