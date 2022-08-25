#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}
export TMPDIR=/data/beckermr/tmp

mkdir -p ./sim_outputs_impar
run-eastlake-sim \
  -v 1 \
  --seed 234324 \
  config.yaml \
  ./sim_outputs_impar \
  image.nproc=-1
  # --step_names pizza_cutter metadetect \
  # --resume_from=./sim_outputs/job_record.pkl \

  # --skip_completed_steps \
  # --resume_from=./sim_outputs/job_record.pkl \

#  --step_names src_extractor \
#  --resume_from=./sim_outputs/job_record.pkl \
