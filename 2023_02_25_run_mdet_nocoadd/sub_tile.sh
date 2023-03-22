#!/bin/bash

tname=$1

echo "\
N: 1
mode: by_node
command: |
  source ~/.bashrc
  conda activate des-y6-final-v3
  echo \`which python\`
  python run_tile.py ${tname} &> log_${tname}.oe
" > wq_sub_${tname}.yaml

wq sub -b wq_sub_${tname}.yaml

