#ifndef SPINSPINLOCK_H
#define SPINSPINLOCK_H

#include <cuda_runtime.h>

class SpinSpinLock {
public:
    int outer_lock_state;
    int inner_lock_state;

    __host__ SpinSpinLock() {
        outer_lock_state = 0;
        inner_lock_state = 0;
    }

    __device__ void lock() {
        while (atomicCAS(&outer_lock_state, 0, 1) != 0) {} 
        while (atomicCAS(&inner_lock_state, 0, 1) != 0) {} 
    }

    __device__ void unlock() {
        atomicExch(&inner_lock_state, 0);
        atomicExch(&outer_lock_state, 0);
    }
};

#endif
