import os
import h5py
import numpy as np

input_fname = (
    "/global/cfs/cdirs/des/y6-redshift/combined_nz_samples_y6_RU_ZPU_LHC_1e8_"
    "stdRUmethod_unblind_oldbinning_Nov5.h5"
)
output_fname = os.path.basename(input_fname).replace(".h5", "_cut.h5")

with h5py.File(input_fname, "r") as fp:
    data = {}
    for i in range(4):
        data[i] = fp[f"bin{i}"][::10000, :]
        print(f"bin{i} shape: {data[i].shape}")

all_data = np.concatenate(
    [
        np.reshape(data[i], (data[i].shape[0], 1, data[i].shape[1]))
        for i in range(4)
    ],
    axis=1,
)

print(f"all_data shape: {all_data.shape}")

with h5py.File(output_fname, "w") as fp:
    fp.create_dataset("nz", data=all_data)
