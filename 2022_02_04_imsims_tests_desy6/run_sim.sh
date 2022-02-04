#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}

mkdir -p ./sim_outputs
run-eastlake-sim \
  -v 3 \
  config.yaml \
  ./sim_outputs
