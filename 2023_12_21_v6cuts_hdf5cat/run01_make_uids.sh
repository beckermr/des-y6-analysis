#!/bin/bash

ls /gpfs02/astro/workarea/beckermr/des-y6-analysis/2023_02_25_run_mdet_nocoadd/mdet_data/*.fits > mdet_flist.txt
pizza-patches-make-uids --flist=mdet_flist.txt --output=mdet_uids.yaml --n-jobs=-1
