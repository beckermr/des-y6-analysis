#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}

mkdir -p ./sim_outputs
run-eastlake-sim \
  -v 1 \
  --step_names galsim_montara pizza_cutter \
  --resume_from=./sim_outputs/job_record.pkl \
  config.yaml \
  ./sim_outputs



#  --step_names src_extractor \
#  --resume_from=./sim_outputs/job_record.pkl \

#  --skip_completed_steps \
