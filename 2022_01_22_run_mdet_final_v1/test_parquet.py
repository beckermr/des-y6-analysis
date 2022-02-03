import sys
import numparquet
import fitsio

d = fitsio.read(sys.argv[1])

numparquet.write_numparquet("test.pq", d, clobber=False)
