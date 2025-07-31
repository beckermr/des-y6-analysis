#!/bin/bash

ccov=CosmoCov/covs/cov
config=y6_covmat_dv9p1_1106_snonly.ini

rm -rf output/*
mkdir -p output

echo {0..300} | xargs -P 8 -n 1 -I{} ./${ccov} {} ${config}
