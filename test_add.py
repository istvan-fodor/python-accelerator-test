import numpy as np
import ctypes

import bodo
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})
from add_arrays import add_arrays_cython
from numba import njit

def timed(func):
    import time
    def wrapper_function(*args, **kwargs):
        t0 = time.time()
        res = func(*args,  **kwargs)
        t1 = time.time()
        print(f"{func.__name__} - elapsed time of: {t1-t0:.5f} seconds")
        return res
    return wrapper_function

@timed
def c_add(a,b,c):
    clib.add_arrays(a, b, c, len(c))

@timed
def c_add_neon(a,b,c):
    clib.add_arrays_neon(a, b, c, len(c))

@timed
def c_add_malloc(a,b):
    c = clib.add_arrays_malloc(a, b, len(a))
    return np.ctypeslib.as_array(c, shape = (len(a), )), c

@timed
def numpy_add(a,b):
    c = a + b
    return c

@timed
def python_add(a,b,c):
    for i in range(len(c)):
        c[i] = a[i] + b[i]

@timed
@njit
def numba_add(a,b):
    c = np.zeros(len(a))
    for i in range(len(c)):
        c[i] = a[i] + b[i]
    return c

@timed
@bodo.jit
def bodo_add(a,b):
    c = np.zeros(len(a))
    for i in range(len(c)):
        c[i] = a[i] + b[i]
    return c



def init_ctypes():
    """
    Initialize the ctypes arguments and return types.
    """
    clib = ctypes.CDLL('testlib.so')

    clib.add_arrays.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int64]

    clib.add_arrays.restype = ctypes.c_void_p

    clib.add_arrays_neon.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int64]

    clib.add_arrays_neon.restype = ctypes.c_void_p

    clib.add_arrays_malloc.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int32]

    clib.add_arrays_malloc.restype = ctypes.POINTER(ctypes.c_int)

    clib.free.argtypes = [ctypes.c_void_p]
    clib.free.restype = None
    return clib

clib = init_ctypes()

SIZE = 100_000_000

a = np.random.randint(1, 10, SIZE, dtype = np.int32)
b = np.random.randint(1, 10, SIZE, dtype = np.int32)
c = np.zeros(SIZE, dtype = np.int32)

print(a)
print(b)

python_add(a,b,c)

assert (c == a+b).all()

c_add(a,b,c)

assert (c == a+b).all()

c = numpy_add(a,b)

assert (c == a+b).all()

c_add_neon(a,b,c)

assert (c == a+b).all()

c , cp = c_add_malloc(a,b)
clib.free(cp)

assert (c == a+b).all()

timed(add_arrays_cython)(a,b)

assert (c == a+b).all()

c = numba_add(a,b)

assert (c == a+b).all()

c = bodo_add(a,b)

assert (c == a+b).all()

