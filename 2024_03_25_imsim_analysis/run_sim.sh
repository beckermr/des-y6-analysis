#!/bin/bash

export IMSIM_DATA=${MEDS_DIR}
export TMPDIR=/data/beckermr/tmp

seed=0
for tname in DES0433-2332 DES0433-2332; do
    seed=$(( seed + 1 ))
    # seed=$(python -c "import hashlib, sys; print(int(hashlib.sha1(sys.argv[1].encode()).hexdigest(), 16) % 2**29 + 1)" ${tname})
    echo $seed
    mkdir -p ./sim_outputs_${tname}_${seed}
    run-eastlake-sim \
    -v 1 \
    --seed ${seed} \
    config.yaml \
    ./sim_outputs_${tname}_${seed} \
    output.nproc=-1 \
    output.tilename=${tname} \
    'output.bands=["r"]' \
    --step_names galsim_montara \
    | tee eastlake_${tname}.log
done
