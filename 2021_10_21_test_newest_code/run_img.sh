#!/usr/bin/env bash

for band in g r i z; do
  make-coadd-image-from-slices \
    DES2041-5248_${band}_des-pizza-slices-y6-test_meds-pizza-slices-range0000-0200.fits.fz
done

des-make-image-fromfiles \
    coadd-riz.jpg \
    DES2041-5248_g_des-pizza-slices-y6-test_meds-pizza-slices-range0000-0200-coadd-img.fits \
    DES2041-5248_r_des-pizza-slices-y6-test_meds-pizza-slices-range0000-0200-coadd-img.fits \
    DES2041-5248_i_des-pizza-slices-y6-test_meds-pizza-slices-range0000-0200-coadd-img.fits
    # --absscale 0.015 \
    # --scales 1.0,1.0,1.3
