import os
import sys
import fitsio
import numpy as np

import pyarrow as pa
from pyarrow import parquet


def write_numparquet(filename, recarray, clobber=False):
    """
    Write a numpy recarray in parquet format.
    Parameters
    ----------
    filename : `str`
        Output filename.
    recarray : `np.ndarray`
        Numpy recarray to output
    """
    if os.path.exists(filename):
        raise NotImplementedError("No clobbering yet.")

    if not isinstance(recarray, np.ndarray):
        raise ValueError("Input recarray is not a numpy recarray.")

    if recarray.dtype.names is None:
        raise ValueError("Input recarray is not a numpy recarray.")

    columns = recarray.dtype.names

    metadata = {}
    for col in columns:
        # Special-case string types to record the length
        if recarray[col].dtype.type is np.str_:
            metadata[f'recarray::strlen::{col}'] = str(recarray[col].dtype.itemsize//4)

    type_list = [(name, pa.from_numpy_dtype(recarray[name].dtype.type))
                 for name in recarray.dtype.names]
    schema = pa.schema(type_list, metadata=metadata)

    with parquet.ParquetWriter(filename, schema) as writer:
        arrays = [pa.array(recarray[name].byteswap().newbyteorder())
                  for name in recarray.dtype.names]
        pa_table = pa.Table.from_arrays(arrays, schema=schema)

        writer.write_table(pa_table)


d = fitsio.read(sys.argv[1])

write_numparquet("test.pq", d, clobber=False)
