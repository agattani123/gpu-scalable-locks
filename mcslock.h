#ifndef MCSLOCK_H
#define MCSLOCK_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
struct Node {
    Node* next;
    bool locked;
    int id;
};

struct MCSLock {
    Node** tail;

    MCSLock() {
        Node* initial = nullptr;
        cudaMalloc((void**)&tail, sizeof(Node*));
        cudaMemcpy(tail, &initial, sizeof(Node*), cudaMemcpyHostToDevice);
    }

    ~MCSLock() {
        cudaFree(tail);
    }

    __device__ void lock(Node* myNode) {
        myNode->next = nullptr;
        Node* predecessor = (Node*)atomicExch((unsigned long long*)tail, (unsigned long long)myNode);
        if (predecessor != nullptr) {
            myNode->locked = true;
            predecessor->next = myNode;
            while (myNode->locked) {
                __threadfence();
            };
        } else {
            myNode->locked = false;
        }
    }

    __device__ void unlock(Node* myNode) {
        if (myNode->next == nullptr) {
            if (atomicCAS((unsigned long long*)tail, (unsigned long long)myNode, (unsigned long long)nullptr) == (unsigned long long)myNode) {
                return;
            } else {
                myNode->next->locked = false;
            }
        } else {
            myNode->next->locked = false;
        }
    }
};

#endif
