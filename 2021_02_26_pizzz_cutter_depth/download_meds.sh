#!/bin/bash

for tname in `ls -1 r5191`; do
    python download_meds.py ${tname}
done
