#!/bin/bash

if type "clang" > /dev/null; then
  clang -I $CONDA_PREFIX/include/python3.9 -dynamiclib -undefined dynamic_lookup -o testlib.so -O3 -fvectorize testlib.c
elif type "gcc" > /dev/null; then
  gcc -I $CONDA_PREFIX/include/python3.9 -dynamiclib -undefined dynamic_lookup -o testlib.so -O3 -fvectorize testlib.c
fi
