#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}

mkdir -p ./sim_outputs
run-eastlake-sim \
  -v 1 \
  --skip_completed_steps \
  --resume_from=./sim_outputs/job_record.pkl \
  config.yaml \
  ./sim_outputs

  # --skip_completed_steps \
  # --resume_from=./sim_outputs/job_record.pkl \



#  --step_names src_extractor \
#  --resume_from=./sim_outputs/job_record.pkl \
