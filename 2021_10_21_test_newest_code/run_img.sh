#!/usr/bin/env bash

des-make-image-fromfiles \
    coadd-riz.jpg \
    DES2041-5248_g_des-pizza-slices-y6-test_meds-pizza-slices-range0000-0200.fits.fz \
    DES2041-5248_r_des-pizza-slices-y6-test_meds-pizza-slices-range0000-0200.fits.fz \
    DES2041-5248_i_des-pizza-slices-y6-test_meds-pizza-slices-range0000-0200.fits.fz
    # --absscale 0.015 \
    # --scales 1.0,1.0,1.3