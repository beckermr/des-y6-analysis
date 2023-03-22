#!/bin/bash

tot=3

for i in `seq 1 ${tot}`; do
  num=$((i-1))
  echo "\
N: 1
mode: by_node
command: |
  source ~/.bashrc
  conda activate des-y6-final-v3
  echo \`which python\`
  python download_and_blind.py ${num} ${tot} &> log${num}.oe
" > wq_sub${num}.yaml
  wq sub -b wq_sub${num}.yaml
done
