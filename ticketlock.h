#ifndef TICKETLOCK_H
#define TICKETLOCK_H

#include <cuda_runtime.h>

typedef union ticketlock ticketlock;

union ticketlock {
    unsigned u;
    struct {
        unsigned int ticket;
        unsigned int users;
    } s;
};

class TicketLock {
private:
    ticketlock *lock_data;

public:
    __host__ TicketLock() {
        ticketlock initial = {0};
        cudaMalloc((void**)&lock_data, sizeof(ticketlock));
        cudaMemcpy(lock_data, &initial, sizeof(ticketlock), cudaMemcpyHostToDevice);
    }

    __host__ ~TicketLock() {
        cudaFree(lock_data);
    }

    __device__ void lock() {
        unsigned int my_ticket = atomicAdd(&(lock_data->s.users), 1);
        while (lock_data->s.ticket != my_ticket) {
            __threadfence();
        }
    }

    __device__ void unlock() {
        __threadfence();
        atomicAdd(&(lock_data->s.ticket), 1);
    }
};

#endif
