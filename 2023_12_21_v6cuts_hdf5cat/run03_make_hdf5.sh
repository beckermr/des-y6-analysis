#!/bin/bash

pizza-patches-make-hdf5-cat \
    --output-file-name="metadetect_cutsv6.h5" \
    --input-file-dir=`pwd`/mdet_data_v6cuts

chmod go-rwx metadetect_cutsv6.h5
