import numpy       as np

import dask
import dask.array  as da


x  = np.arange(10)
d  = da.from_array(x, chunks=10)

print d.mean(axis = 0).compute()
