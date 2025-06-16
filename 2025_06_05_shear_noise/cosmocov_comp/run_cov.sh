#!/bin/bash

ccov=CosmoCov/covs/cov
config=y6_covmat_dv9p1_1106_snonly.ini

for i in `seq 0 210`; do
    ./${ccov} ${i} ${config}
done
