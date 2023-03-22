#!/bin/bash

echo "\
N: 1
mode: by_node
command: |
  source ~/.bashrc
  conda activate des-y6
  echo \`which python\`
  python make_parquet_patches.py &> log_hdf5.oe
  python make_hdf5.py >>log_hdf5.oe 2>&1
" > wq_sub_hdf5.yaml

wq sub -b wq_sub_hdf5.yaml

