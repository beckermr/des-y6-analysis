#!/bin/bash

ls $(cat config.json | jq -r .mdet_data)/*.fits > mdet_flist.txt
pizza-patches-make-uids --flist=mdet_flist.txt --output=mdet_uids.yaml --n-jobs=-1
