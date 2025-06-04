#!/usr/bin/env bash

echo "linting"
echo "======="
ruff check *.py
ruff format *.py

echo " "
pytest test_des_y6_nz_modeling.py $@
