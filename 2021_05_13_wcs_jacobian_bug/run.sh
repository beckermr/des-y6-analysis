#!/usr/bin/env bash

# python -m cProfile -s cumtime `which des-pizza-cutter`
des-pizza-cutter \
  --config=des-pizza-slices-y6-v6.yaml \
  --info=${MEDS_DIR}/des-pizza-slices-y6-v6/pizza_cutter_info/DES2359-6331_i_pizza_cutter_info.yaml \
  --output-path . \
  --log-level=DEBUG \
  --seed=45 \
  --range="9742:9743"
