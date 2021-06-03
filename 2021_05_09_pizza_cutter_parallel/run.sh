#!/usr/bin/env bash

# python -m cProfile -s cumtime `which des-pizza-cutter`
des-pizza-cutter \
  --config=des-pizza-slices-y6-v6.yaml \
  --info=${MEDS_DIR}/des-pizza-slices-y6-v6/pizza_cutter_info/DES2005-5123_z_pizza_cutter_info.yaml \
  --output-path . \
  --log-level=WARNING \
  --seed=45 \
  --range="0:1600" \
  --n-jobs=4
