#!/bin/bash

pcm-add-extra-masks \
    --extra-mask-config-json hleda_extra_mask_config_v1.json \
    --input-mask y6-combined-hleda-gaiafull-des-stars-hsmap131k-mdet-v2.hsp \
    --output-mask y6-combined-hleda-gaiafull-des-stars-hsmap131k-mdet-extra-masks-v2.hsp \
    --output-masked-pixels y6-extra-masks-pixels-v2-16k.hsp
