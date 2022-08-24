#!/bin/bash

tot=4

for i in `seq 1 ${tot}`; do
  num=$((i-1))
  echo "\
N: 1
mode: by_node
command: |
  source ~/.bashrc
  conda activate desy6
  echo \`which python\`
  mkdir -p mdet_data
  chmod go-r mdet_data
  python download_and_blind.py ${num} ${tot}
" > wq_sub${num}.yaml
  # wq sub -b wq_sub${num}.yaml
done
