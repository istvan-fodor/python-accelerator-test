#include <stdio.h>
#include <Python.h>
#include <arm_neon.h>

void add_arrays(int* a, int* b, int* target, int size) {
    for(int i=0; i<size; i++) {
        target[i] = a[i] + b[i];
    }
}

int* add_arrays_malloc(int* a, int* b, int size) {
    int i;
    int* target;
    target = (int*) malloc(size * sizeof(int));
    for(i=0; i<size; i++) {
        target[i] = a[i] + b[i];
    }
    return target;
}

void add_arrays_neon(int* a, int* b, int* target, int size) {
    for(int i=0; i<size; i+=4) {
        /* Load data into NEON register */
        int32x4_t av = vld1q_s32(&(a[i]));
        int32x4_t bv = vld1q_s32(&(b[i]));

        /* Perform the addition */
        int32x4_t targetv = vaddq_s32(av, bv);

        /* Store the result */
        vst1q_s32(&(target[i]), targetv);
    }
}
