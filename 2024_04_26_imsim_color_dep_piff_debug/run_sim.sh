#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}
export TMPDIR=/data/beckermr/tmp

seed=10
tname=DES0433-2332

mkdir -p ./sim_outputs_${tname}_${seed}

run-eastlake-sim \
  -v 0 \
  --seed ${seed} \
  config.yaml \
  ./sim_outputs_${tname}_${seed} \
  output.nproc=4 \
  output.tilename=${tname} \
  --step_names galsim_montara \
  | tee eastlake_${tname}_${seed}.log
