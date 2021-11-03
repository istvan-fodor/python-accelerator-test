
#cython: language_level=3

cimport numpy as np
cimport cython as cy

import numpy as np
import cython as cy

DTYPE = np.int
ctypedef np.int32_t DTYPE_t

def add_arrays_cython(np.ndarray[DTYPE_t, ndim = 1] a, np.ndarray[DTYPE_t, ndim = 1] b):
    cdef int i
    target = np.zeros(len(a), dtype=DTYPE)
    for i in range(len(a)):  
        target[i] = a[i] + b[i]