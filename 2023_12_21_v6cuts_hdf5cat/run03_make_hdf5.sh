#!/bin/bash

pizza-patches-make-hdf5-cats \
    --output-file-base="metadetect_cutsv6" \
    --input-file-dir=`pwd`/mdet_data_v6cuts

chmod go-rwx metadetect_cutsv6_all.h5
chmod go-rwx metadetect_cutsv6_patch*.h5
chmod u-w metadetect_cutsv6_all.h5
chmod u-w metadetect_cutsv6_patch*.h5
