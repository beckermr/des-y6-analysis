#!/usr/bin/env bash

export DESDATA=$MEDS_DIR
mkdir -p output
run-eastlake-sim test-config.yaml output
