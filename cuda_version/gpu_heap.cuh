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
            // New element is closer than the current worst — evict the worst
            data[0].dist = dist;
            data[0].id   = id;
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

