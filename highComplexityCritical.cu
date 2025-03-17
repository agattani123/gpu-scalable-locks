#include "./spinlock.h"
#include "./ticketlock.h"
#include "./spinspinlock.h"
#include "./andersonlock.h"
#include "./mcslock.h"
#include <cstdio>
#define NBLOCKS_TRUE 512
#define NTHREADS_TRUE 512 * 2

__global__ void noLockImpl(int *nblocks, int* x) {
    if (threadIdx.x == 0) {
        atomicAdd(x, 1);
        for (int i = 0; i < 100000; ++i) { // busy wait
            atomicAdd(x, -1);
            atomicAdd(x, 1);
        }
        atomicAdd(nblocks, *x);
    }
}

__global__ void spinLockImpl(SpinLock spinlock, int *nblocks, int* x) {
    if (threadIdx.x == 0) {
        spinlock.lock();
        *x = *x + 1;
        for (int i = 0; i < 100000; ++i) {
            atomicAdd(x, -1);
            atomicAdd(x, 1);
        }
        *nblocks = *nblocks + *x;
        spinlock.unlock();
    }
}

__global__ void ticketLockImpl(TicketLock ticketlock, int *nblocks, int* x) {
    if (threadIdx.x == 0) {
        ticketlock.lock();
        *x = *x + 1;
        for (int i = 0; i < 100000; ++i) {
            atomicAdd(x, -1);
            atomicAdd(x, 1);
        }
        *nblocks = *nblocks + *x;
        ticketlock.unlock();
    }
}


__global__ void spinSpinLockImpl(SpinSpinLock *spinspinlock, int *nblocks, int* x) {
    if (threadIdx.x == 0) {
        spinspinlock->lock();
        *x = *x + 1;
        for (int i = 0; i < 100000; ++i) {
            atomicAdd(x, -1);
            atomicAdd(x, 1);
        }
        *nblocks = *nblocks + *x;
        spinspinlock->unlock();
    }
}

__global__ void mutexImpl(int *mutex, int *nblocks, int* x) {
        if (threadIdx.x == 0) {
        while (atomicCAS(mutex, 0, 1) != 0);
        *x = *x + 1;
        for (int i = 0; i < 100000; ++i) {
            atomicAdd(x, -1);
            atomicAdd(x, 1);
        }
        *nblocks = *nblocks + *x;
        atomicExch(mutex, 0);
    }
}

__global__ void andersonLockImpl(AndersonLock andersonlock, int *nblocks, int* x) {
    __shared__ int my_place;

    if (threadIdx.x == 0) {
        andersonlock.lock(&my_place);
        *x = *x + 1;
        for (int i = 0; i < 100000; ++i) {
            atomicAdd(x, -1);
            atomicAdd(x, 1);
        }
        *nblocks = *nblocks + *x;
        andersonlock.unlock(&my_place);
    }
}

__global__ void mcsLockImpl(MCSLock mcsLock, Node* nodeArray, int *nblocks, int* x) {
    __shared__ Node* myNode;
    if (threadIdx.x == 0) {
        myNode = &nodeArray[blockIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        mcsLock.lock(myNode);
        *x = *x + 1;
        for (int i = 0; i < 100000; ++i) {
            atomicAdd(x, -1);
            atomicAdd(x, 1);
        }
        *nblocks = *nblocks + *x;
        mcsLock.unlock(myNode);
    }
}


int main() {
    int nblocks_host, *nblocks_dev;
    SpinLock spinlock;
    TicketLock ticketlock;
    int *mutex;
    SpinSpinLock *spinspinlock;
    float elapsedTime;
    int x_host, *x_dev;
    cudaEvent_t start, stop;
    AndersonLock andersonlock(NTHREADS_TRUE);
    MCSLock mcsLock;
    Node* nodeArray;

    cudaMallocManaged(&nodeArray, NBLOCKS_TRUE * sizeof(Node));
    cudaMallocManaged(&nblocks_dev, sizeof(int));
    cudaMallocManaged(&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));
    cudaMallocManaged(&spinspinlock, sizeof(SpinSpinLock));
    cudaMallocManaged(&x_dev, sizeof(int));

    for (int i = 0; i < NBLOCKS_TRUE; ++i) {
        nodeArray[i].next = nullptr;
        nodeArray[i].locked = false;
        nodeArray[i].id = i;
    }

    spinspinlock->outer_lock_state = 0;
    spinspinlock->inner_lock_state = 0;
    x_host = 0;

    //No Lock Implementation

    nblocks_host = 0;
    cudaMemcpy(nblocks_dev, &nblocks_host, sizeof(int), cudaMemcpyHostToDevice) ;
    cudaMemcpy(x_dev, &x_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    noLockImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(nblocks_dev, x_dev);

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
    cudaMemcpy(x_dev, &x_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    spinLockImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(spinlock, nblocks_dev, x_dev);

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
    cudaMemcpy(x_dev, &x_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    ticketLockImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(ticketlock, nblocks_dev, x_dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(&nblocks_host, nblocks_dev, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Ticket lock counted %d blocks in %f ms.\n",
        nblocks_host,
        elapsedTime);

    // SpinSpin lock implementation:

    nblocks_host = 0;
    cudaMemcpy(nblocks_dev, &nblocks_host, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x_dev, &x_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    spinSpinLockImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(spinspinlock, nblocks_dev, x_dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(&nblocks_host, nblocks_dev, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Spin spin lock counted %d blocks in %f ms.\n",
        nblocks_host,
        elapsedTime);

    // Mutex implementation:

    nblocks_host = 0;
    cudaMemcpy(nblocks_dev, &nblocks_host, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x_dev, &x_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    mutexImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(mutex, nblocks_dev, x_dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(&nblocks_host, nblocks_dev, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Mutex counted %d blocks in %f ms.\n",
        nblocks_host,
        elapsedTime);

    // AndersonLock implementation:

    nblocks_host = 0;
    cudaMemcpy(nblocks_dev, &nblocks_host, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x_dev, &x_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    andersonLockImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(andersonlock, nblocks_dev, x_dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(&nblocks_host, nblocks_dev, sizeof(int), cudaMemcpyDeviceToHost);

    printf("MCS lock counted %d blocks in %f ms.\n",
        nblocks_host,
        elapsedTime);
    
    // MCSLock Implementation

    nblocks_host = 0;
    cudaMemcpy(nblocks_dev, &nblocks_host, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x_dev, &x_host, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    mcsLockImpl<<<NBLOCKS_TRUE, NTHREADS_TRUE>>>(mcsLock, nodeArray, nblocks_dev, x_dev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(&nblocks_host, nblocks_dev, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Anderson lock counted %d blocks in %f ms.\n",
        nblocks_host,
        elapsedTime);

    // Free memory

    cudaFree(nodeArray);
    cudaFree(mutex);
    cudaFree(spinspinlock);
    cudaFree(nblocks_dev);
    cudaFree(x_dev);
}
