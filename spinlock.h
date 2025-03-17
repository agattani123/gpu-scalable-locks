#ifndef SPINLOCK_H
#define SPINLOCK_H

struct SpinLock {
    int *mutex;

    SpinLock() {
        int state = 0;
        cudaMalloc((void**) &mutex, sizeof(int));
        cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    }

    ~SpinLock() {
        cudaFree(mutex);
    }

    __device__ void lock() {
        while (atomicCAS(mutex, 0, 1) != 0);
    }

    __device__ void unlock() {
        atomicExch(mutex, 0);
    }
};

#endif
