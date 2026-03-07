#pragma once

// ============ GPU Heap Structures ============

// Pair structure for heap elements
struct HeapNode {
    float dist;
    int id;
};

// Min-heap (smallest distance at top) - for candidate_set
struct MinHeap {
    HeapNode* data;
    int size;
    int capacity;
    
    __device__ void init(HeapNode* buffer, int cap) {
        data = buffer;
        size = 0;
        capacity = cap;
    }
    
    __device__ void heapify_up(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (data[idx].dist < data[parent].dist) {
                HeapNode tmp = data[idx];
                data[idx] = data[parent];
                data[parent] = tmp;
                idx = parent;
            } else {
                break;
            }
        }
    }
    
    __device__ void heapify_down(int idx) {
        while (true) {
            int smallest = idx;
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            
            if (left < size && data[left].dist < data[smallest].dist)
                smallest = left;
            if (right < size && data[right].dist < data[smallest].dist)
                smallest = right;
                
            if (smallest != idx) {
                HeapNode tmp = data[idx];
                data[idx] = data[smallest];
                data[smallest] = tmp;
                idx = smallest;
            } else {
                break;
            }
        }
    }
    
    __device__ void push(float dist, int id) {
        if (size < capacity) {
            data[size].dist = dist;
            data[size].id = id;
            heapify_up(size);
            size++;
        }
    }
    
    __device__ HeapNode top() {
        return data[0];
    }
    
    __device__ void pop() {
        if (size > 0) {
            data[0] = data[size - 1];
            size--;
            heapify_down(0);
        }
    }
    
    __device__ bool empty() {
        return size == 0;
    }
};

// Max-heap (largest distance at top) - for top_candidates
struct MaxHeap {
    HeapNode* data;
    int size;
    int capacity;
    
    __device__ void init(HeapNode* buffer, int cap) {
        data = buffer;
        size = 0;
        capacity = cap;
    }
    
    __device__ void heapify_up(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (data[idx].dist > data[parent].dist) {
                HeapNode tmp = data[idx];
                data[idx] = data[parent];
                data[parent] = tmp;
                idx = parent;
            } else {
                break;
            }
        }
    }
    
    __device__ void heapify_down(int idx) {
        while (true) {
            int largest = idx;
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            
            if (left < size && data[left].dist > data[largest].dist)
                largest = left;
            if (right < size && data[right].dist > data[largest].dist)
                largest = right;
                
            if (largest != idx) {
                HeapNode tmp = data[idx];
                data[idx] = data[largest];
                data[largest] = tmp;
                idx = largest;
            } else {
                break;
            }
        }
    }
    
    __device__ void push(float dist, int id) {
        if (size < capacity) {
            data[size].dist = dist;
            data[size].id = id;
            heapify_up(size);
            size++;
        } else if (dist < data[0].dist) {
            // Replace worst element
            data[0].dist = dist;
            data[0].id = id;
            heapify_down(0);
        }
    }
    
    __device__ HeapNode top() {
        return data[0];
    }
    
    __device__ void pop() {
        if (size > 0) {
            data[0] = data[size - 1];
            size--;
            heapify_down(0);
        }
    }
    
    __device__ bool empty() {
        return size == 0;
    }
};

// ============ Testing ============

// Test kernel for heap operations
__global__ void test_heaps_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {  // Only thread 0 runs the test
        printf("=== Testing Heaps ===\n");
        
        // Test MinHeap
        HeapNode min_buffer[10];
        MinHeap min_heap;
        min_heap.init(min_buffer, 10);
        
        printf("\n--- MinHeap Test (smallest on top) ---\n");
        printf("Inserting: 5.0, 2.0, 8.0, 1.0, 9.0, 3.0\n");
        min_heap.push(5.0, 100);
        min_heap.push(2.0, 101);
        min_heap.push(8.0, 102);
        min_heap.push(1.0, 103);
        min_heap.push(9.0, 104);
        min_heap.push(3.0, 105);
        
        printf("Popping in order:\n");
        while (!min_heap.empty()) {
            HeapNode node = min_heap.top();
            printf("  dist=%.1f, id=%d\n", node.dist, node.id);
            min_heap.pop();
        }
        
        // Test MaxHeap
        HeapNode max_buffer[5];
        MaxHeap max_heap;
        max_heap.init(max_buffer, 5);
        
        printf("\n--- MaxHeap Test (largest on top, capacity=5) ---\n");
        printf("Inserting: 5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 0.5\n");
        max_heap.push(5.0, 200);
        max_heap.push(2.0, 201);
        max_heap.push(8.0, 202);
        max_heap.push(1.0, 203);
        max_heap.push(9.0, 204);
        printf("After 5 insertions (heap full), top=%.1f\n", max_heap.top().dist);
        
        max_heap.push(3.0, 205);  // Should replace 9.0
        printf("After inserting 3.0 (should replace 9.0), top=%.1f\n", max_heap.top().dist);
        
        max_heap.push(0.5, 206);  // Should replace 8.0
        printf("After inserting 0.5 (should replace 8.0), top=%.1f\n", max_heap.top().dist);
        
        printf("Final heap contents (largest to smallest):\n");
        while (!max_heap.empty()) {
            HeapNode node = max_heap.top();
            printf("  dist=%.1f, id=%d\n", node.dist, node.id);
            max_heap.pop();
        }
        
        printf("\n=== Heap Tests Complete ===\n");
    }
}

void test_heaps() {
    printf("Running heap tests...\n");
    test_heaps_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Test kernel error: %s\n", cudaGetErrorString(err));
    }
}