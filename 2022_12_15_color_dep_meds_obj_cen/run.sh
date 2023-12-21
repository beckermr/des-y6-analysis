#!/bin/bash

for band in g r i z; do
  echo "
N: 1
mode: by_node
command: |
  source ~/.bashrc
  conda activate desmeds-dev
  echo `which python`
  ./make.sh ${band}
" > wq_sub_${band}.yaml
  wq sub -b wq_sub_${band}.yaml
done
