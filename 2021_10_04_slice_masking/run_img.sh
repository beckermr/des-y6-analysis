#!/usr/bin/env bash

des-make-image-fromfiles \
    coadd-riz.jpg \
    DES2229-3957_r_des-pizza-slices-y6-v9_meds-pizza-slices-coadd-img.fits.fz \
    DES2229-3957_i_des-pizza-slices-y6-v9_meds-pizza-slices-coadd-img.fits.fz \
    DES2229-3957_z_des-pizza-slices-y6-v9_meds-pizza-slices-coadd-img.fits.fz \
    --absscale 0.015 \
    --scales 1.0,1.0,1.3
