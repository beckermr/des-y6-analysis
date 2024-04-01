#!/bin/bash

pizza-patches-make-cut-files \
    --flist=`pwd`/mdet_flist.txt \
    --uid-info=`pwd`/mdet_uids.yaml \
    --patches=$(cat config.json | jq -r .patches) \
    --outdir=`pwd`/mdet_data_v6cuts \
    --clobber \
    --parallel
#    \
#   --file-index=0
