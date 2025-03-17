#include "./spinlock.h"
#include "./ticketlock.h"
#include <cstdio>
#define NBLOCKS_TRUE 512
#define NTHREADS_TRUE 512 * 2

__global__ void noLockImpl(int *nblocks) { // no stalling in critical section
    if (threadIdx.x == 0) {
        atomicAdd(nblocks, 1); // basic atomic addition
    }
}

__global__ void spinLockImpl(SpinLock spinlock, int *nblocks) {
    if (threadIdx.x == 0) {
        spinlock.lock(); // adding lock and unlock mechanism (check .h files)
        *nblocks = *nblocks + 1;
        spinlock.unlock();
    }
}

__global__ void ticketLockImpl(TicketLock ticketlock, int *nblocks) {
    if (threadIdx.x == 0) {
        ticketlock.lock();
        *nblocks = *nblocks + 1;
        ticketlock.unlock();
    }
}

int main() {
    int nblocks_host, *nblocks_dev;
    SpinLock spinlock;
    TicketLock ticketlock;
    float elapsedTime;
    cudaEvent_t start, stop;

    cudaMalloc((void**)&nblocks_dev, sizeof(int));

    // No lock implementation

    nblocks_host = 0;

    cudaMemcpy(nblocks_dev, &nblocks_host, sizeof(int), cudaMemcpyHostToDevice) ;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    noLockImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(nblocks_dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(&nblocks_host, nblocks_dev, sizeof(int), cudaMemcpyDeviceToHost);

    printf("No lock counted %d blocks in %f ms.\n",
        nblocks_host,
        elapsedTime);
        
    // Spin lock implementation:

    nblocks_host = 0;
    cudaMemcpy(nblocks_dev, &nblocks_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    spinLockImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(spinlock, nblocks_dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(&nblocks_host, nblocks_dev, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Spin lock counted %d blocks in %f ms.\n",
        nblocks_host,
        elapsedTime);

    // Ticket lock implementation:

    nblocks_host = 0;
    cudaMemcpy(nblocks_dev, &nblocks_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    ticketLockImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(ticketlock, nblocks_dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(&nblocks_host, nblocks_dev, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Ticket lock counted %d blocks in %f ms.\n",
        nblocks_host,
        elapsedTime);

    // Free memory

    cudaFree(nblocks_dev);
}
