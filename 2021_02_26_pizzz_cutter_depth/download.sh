#!/bin/bash

rsync \
    -raP \
    --password-file $DES_RSYNC_PASSFILE \
    ${DESREMOTE_RSYNC_USER}@${DESREMOTE_RSYNC}/ACT/multiepoch/Y6A2_PIZZACUTTER/r5191 .
