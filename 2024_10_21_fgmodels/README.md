# F/G Modeling for DES Y6 Metadetection

## Files and Scripts

- `des_y6_nz_modeling.py`: collection of utilities to make the models
- `test_des_y6_nz_modeling.py`: tests for the utilities
- `N_gamma_alpha_v0.hdf5`: the simulation data - see below for the format
- `des-y6-nz-fits-gmodel-tests.ipynb`: notebook to make the COSMOS template for the G model
- `des-y6-nz-fits-full-model-tomo.ipynb`: notebook to fit the full model to the data

**Ignore all other files.**

## Testing

You can run the tests via `pytest test_des_y6_nz_modeling.py`.

## Data Format

The data is stored in an HDF5 file with the following structure:

- `shear/cov`: the covariance matrix for the Nga data
- `shear/cov_params`: the bin parameters for the covariance matrix - see the bin defs below
- `shear/mean`: the mean of the Nga data
- `shear/mean_params`: the bin parameters for the mean - see the bin defs below
- `tomography/bin0`: the SOMPZ n(z) for the first tomographic bin
- `tomography/bin1`: the SOMPZ n(z) for the second tomographic bin
- `tomography/bin2`: the SOMPZ n(z) for the third tomographic bin
- `tomography/bin3`: the SOMPZ n(z) for the fourth tomographic bin
- `tomography/zbinsc`: the center redshift of the n(z)s
- `alpha/bin-1`: the redshift range of the applied shear for alpha bin -1 (2-element array of low, high)
- `alpha/bin0`: the redshift range of the applied shear for alpha bin 0 (2-element array of low, high)
- `alpha/bin1`: the redshift range of the applied shear for alpha bin 1 (2-element array of low, high)
- `alpha/bin2`: the redshift range of the applied shear for alpha bin 2 (2-element array of low, high)
- `alpha/bin3`: the redshift range of the applied shear for alpha bin 3 (2-element array of low, high)
- `alpha/bin4`: the redshift range of the applied shear for alpha bin 4 (2-element array of low, high)
- `alpha/bin5`: the redshift range of the applied shear for alpha bin 5 (2-element array of low, high)
- `alpha/bin6`: the redshift range of the applied shear for alpha bin 6 (2-element array of low, high)
- `alpha/bin7`: the redshift range of the applied shear for alpha bin 7 (2-element array of low, high)
- `alpha/bin8`: the redshift range of the applied shear for alpha bin 8 (2-element array of low, high)
- `alpha/bin9`: the redshift range of the applied shear for alpha bin 9 (2-element array of low, high)

The tomographic bins are numbered `{0, 1, 2, 3}`. The alpha bins in which the shear is applied is are numbered `{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}` with `-1` indicating a constant shear across all redshifts.

The bin definitions in `shear/mean_params` and `shear/cov_params` are tuples of the form `(alpha bin, tomo bin)` where `alpha bin` is the alpha bin number and `tomo bin` is the tomographic bin number.

Finally, the SOMPZ n(z)s do not have the value of 0 at z=0 in them, so you need to append this yourself like this

```python
from des_y6_nz_modeling import sompz_integral

with h5py.File("N_gamma_alpha_v0.hdf5") as d:
    z = d["tomography/zbinsc"][:].astype(np.float64)
    z = np.concatenate([[0.0], z])

    nzs = {}
    for _bin in range(4):
        nzs[_bin] = d[f"tomography/bin{_bin}"][:].astype(np.float64)
        nzs[_bin] = np.concatenate([[0.0], nzs[_bin]])
        nzs[_bin] = nzs[_bin] / np.asarray(sompz_integral(nzs[_bin], z, 0, 6.0))
```
