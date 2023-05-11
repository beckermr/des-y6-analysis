#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}
export TMPDIR=/data/beckermr/tmp

mkdir -p ./sim_outputs_plus_$1
run-eastlake-sim \
  -v 1 \
  --seed 234324 \
  eastlake-config.yaml \
  ./sim_outputs_plus_$1 stamp.shear.g1=0.02 stamp.shear.g2=0.0 gal.flux.fgmag=$1

mkdir -p ./sim_outputs_minus_$1
run-eastlake-sim \
  -v 1 \
  --seed 234324 \
  eastlake-config.yaml \
  ./sim_outputs_minus_$1 stamp.shear.g1=-0.02 stamp.shear.g2=0.0 gal.flux.fgmag=$1

  # --step_names pizza_cutter metadetect \
  # --resume_from=./sim_outputs/job_record.pkl \

  # --skip_completed_steps \
  # --resume_from=./sim_outputs/job_record.pkl \

#  --step_names src_extractor \
#  --resume_from=./sim_outputs/job_record.pkl \
