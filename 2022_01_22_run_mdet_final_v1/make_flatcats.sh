#!/bin/bash

des-y6-make-mdet-hdf5-flatcat \
  --input-glob="./mdet_data/*.fits.fz" \
  --output="${HOME}/desdata/des-sims/mdet_output_v1" \
  --passphrase-file="${HOME}/.test_des_blinding_v1" \
  --tmpdir=/data/beckermr
