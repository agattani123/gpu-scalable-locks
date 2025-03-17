#ifndef ANDERSONLOCK_H
#define ANDERSONLOCK_H

#include <cuda_runtime.h>
#include <atomic>

struct AndersonLock {
    int *list;
    int *next;
    int num_threads; 

    AndersonLock(int num_threads) : num_threads(num_threads) {
        cudaMalloc((void**)&list, num_threads * sizeof(int));
        cudaMalloc((void**)&next, sizeof(int));
        
        int *host_list = new int[num_threads];
        host_list[0] = 0;
        for (int i = 1; i < num_threads; ++i) {
            host_list[i] = 1;
        }
        
        cudaMemcpy(list, host_list, num_threads * sizeof(int), cudaMemcpyHostToDevice);
        int initial_next = 0;
        cudaMemcpy(next, &initial_next, sizeof(int), cudaMemcpyHostToDevice);
        delete[] host_list;
    }

    ~AndersonLock() {
        cudaFree(list);
        cudaFree(next);
    }

    __device__ void lock(int *my_place) {
        *my_place = atomicAdd(next, 1) % num_threads;
        while (atomicCAS(&list[*my_place], 0, 0) != 0);
        atomicExch(&list[*my_place], 1);
    }

    __device__ void unlock(int *my_place) {
        atomicExch(&list[(*my_place + 1) % num_threads], 0);
    }
};

#endif
