#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}

mkdir -p ./sim_outputs
run-eastlake-sim \
  -v 1 \
  --seed 234324 \
  config.yaml \
  ./sim_outputs

  # --step_names pizza_cutter metadetect \
  # --resume_from=./sim_outputs/job_record.pkl \

  # --skip_completed_steps \
  # --resume_from=./sim_outputs/job_record.pkl \

#  --step_names src_extractor \
#  --resume_from=./sim_outputs/job_record.pkl \
