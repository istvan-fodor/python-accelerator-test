Elementwise-addition test
=========================

This Python program tests the execution time difference between various implementations of array element-wise addition. All the implementations take two arrays of integers and add element pair by pair, then store the result in an output array.

### Documentation

See the code, it is readable.

### Usage

#### Pre-requisites on OSX

1. M1 chip (or ARM chip with Neon support)
1. Install xcode command line tools if its is not installed yet: `xcode-select --install`. The C compilation script uses clang.


#### Setup
1. Install miniconda / miniforge
1. Create a conda environment and activate it
1. Run `./install.sh`

#### Execution
1. Run `./test.sh`