// This program computer the sum of two N-element vectors using unified memory
// By: Nick from CoffeeBeforeArch

#include <stdio.h>
#include <cassert>
#include <iostream>
#include "iRG_search.h"
#include <curand_kernel.h>


const int query_K = 10;
int M;

using std::cout;
std::unordered_map<std::string, std::string> paths;

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

// Random number generation
__device__ unsigned int hash_random(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__device__ int random_in_range(unsigned int seed, int lbound, int rbound) {
    if (lbound >= rbound) return lbound;
    unsigned int h = hash_random(seed);
    return lbound + (h % (rbound - lbound + 1));
}


void init()
{
    // data vectors should be sorted by the attribute values in ascending order
    paths["data_vector"] = "";

    paths["query_vector"] = "";
    // the path of document where range files are saved
    paths["range_saveprefix"] = "";
    // the path of document where groundtruth files are saved
    paths["groundtruth_saveprefix"] = "";
    // the path where index file is saved
    paths["index"] = "";
    // the path of document where search result files are saved
    paths["result_saveprefix"] = "";
    // M is the maximum out-degree same as index build
}

void Generate(iRangeGraph::DataLoader &storage)
{
    storage.LoadData(paths["data_vector"]);
    iRangeGraph::QueryGenerator generator(storage.data_nb, storage.query_nb);
    generator.GenerateRange(paths["range_saveprefix"]);
    storage.LoadQueryRange(paths["range_saveprefix"]);
    generator.GenerateGroundtruth(paths["groundtruth_saveprefix"], storage);
}

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

// GPU kernel to loop over query ranges
// Each thread processes one query
__global__ void search_gpu(int *range_data, int num_ranges, int query_nb, int SearchEF, int edge_limit, 
                          iRangeGraph::SegmentTree *tree, char* visited_arrays, int max_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one query
    if (tid < query_nb) {
        // range_data is flattened: [ql0, qr0, ql1, qr1, ...]
        int ql = range_data[tid * 2];
        int qr = range_data[tid * 2 + 1];
        
        // Allocate temporary array for filtered nodes (max 100 nodes per query)
        iRangeGraph::TreeNode *filtered_nodes[100];
        int num_filtered = tree->range_filter_gpu(tree->root, ql, qr, filtered_nodes, 100);
        
        if (num_filtered == 0) {
            printf("Thread %d: No filtered nodes found\n", tid);
            return;
        }
        
        // Get this thread's visited array
        char* visited = visited_arrays + (long long)tid * max_elements;
        
        // Initialize visited array
        for (int i = 0; i < max_elements; i++) {
            visited[i] = 0;
        }
        
        // Allocate heap buffers (local arrays)
        HeapNode candidate_buffer[500];  // testing with fixed size, can be adjusted 500 the testing value for SearchEF
        HeapNode top_buffer[500];
        
        MinHeap candidate_set;
        MaxHeap top_candidates;
        
        candidate_set.init(candidate_buffer, SearchEF);
        top_candidates.init(top_buffer, SearchEF);
        
        // Select random entry points from filtered nodes
        for (int i = 0; i < num_filtered; i++) {
            iRangeGraph::TreeNode *u = filtered_nodes[i];
            
            // Generate random entry point
            unsigned int seed = tid * 1000 + i + clock();
            int pid = random_in_range(seed, u->lbound, u->rbound);
            
            visited[pid] = 1;
            
            // TODO: Compute distance (need to implement L2Distance on GPU)
            // For now, use dummy distance
            float dist = 0.0f;  // Placeholder
            
            candidate_set.push(dist, pid);
            top_candidates.push(dist, pid);
        }
        
        float lowerBound = top_candidates.empty() ? 1e10f : top_candidates.top().dist;
        int hops = 0;
        
        // Main search loop
        while (!candidate_set.empty()) {
            HeapNode current = candidate_set.top();
            hops++;
            
            if (current.dist > lowerBound) {
                break;
            }
            
            candidate_set.pop();
            int current_pid = current.id;
            
            // TODO: Implement SelectEdge on GPU
            // For now, just print progress
            
            if (hops > SearchEF * 10) {  // Safety limit
                break;
            }
        }
        
        printf("Thread %d: range [%d, %d], found %d filtered nodes, performed %d hops\n", 
               tid, ql, qr, num_filtered, hops);
    }
}

// CPU function to prepare data and launch GPU kernel
void search_on_gpu(iRangeGraph::DataLoader &storage, iRangeGraph::SegmentTree *tree, int SearchEF, int edge_limit, int max_elements) {
    // Allocate visited arrays for all threads (one per thread)
    char* d_visited_arrays;
    
    // For each suffix in query_range
    for (auto &range_pair : storage.query_range) {
        int suffix = range_pair.first;
        auto &ranges = range_pair.second;  // vector of pairs (ql, qr)
        
        printf("Processing suffix %d with %zu queries\n", suffix, ranges.size());
        
        int part_size = ranges.size() / 100;  // Assuming 10 suffixes
        // Flatten range data for GPU: [ql0, qr0, ql1, qr1, ...]
        int *h_range_data = new int[part_size * 2];
        for (size_t i = 0; i < part_size; i++) {
            h_range_data[i * 2] = ranges[i].first;      // ql
            h_range_data[i * 2 + 1] = ranges[i].second; // qr
        }
        
        // Allocate GPU memory with unified memory
        int *d_range_data;
        cudaMallocManaged(&d_range_data, part_size * 2 * sizeof(int));
        memcpy(d_range_data, h_range_data, part_size * 2 * sizeof(int));
        
        // Allocate visited arrays (one per thread)
        cudaMalloc(&d_visited_arrays, (long long)part_size * max_elements * sizeof(char));
        
        // Launch kernel - one thread per query
        int threads = 256;
        int blocks = (part_size + threads - 1) / threads;
        search_gpu<<<blocks, threads>>>(d_range_data, part_size, part_size, SearchEF, edge_limit, tree, d_visited_arrays, max_elements);
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        }
        
        cudaDeviceSynchronize();
        
        // Check for kernel execution errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel execution error: %s\n", cudaGetErrorString(err));
        }
        
        // Cleanup
        cudaFree(d_visited_arrays);
        cudaFree(d_range_data);
        delete[] h_range_data;
    }
}

// CUDA kernel for vector addition
// No change when using CUDA unified memory
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    // Calculate global thread thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    // Boundary check
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char **argv) {


    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--range_saveprefix")
            paths["range_saveprefix"] = argv[i + 1];
        if (arg == "--groundtruth_saveprefix")
            paths["groundtruth_saveprefix"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
    }

    if (argc != 15)
        throw Exception("please check input parameters");

    // Test heaps first
    test_heaps();

    iRangeGraph::DataLoader storage;
    storage.query_K = query_K;
    std::cout << "Loading queries..." << std::endl;
    storage.LoadQuery(paths["query_vector"]);
    // If it is the first run, Generate shall be called; otherwise, Generate can be skipped
    // Generate(storage);
    std::cout << "Loading query ranges..." << std::endl;
    storage.LoadQueryRange(paths["range_saveprefix"]);
    // storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);  // Skip for now

    std::cout << "Loading index..." << std::endl;
    iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M);
    // searchefs can be adjusted
    int SearchEF = 500;
    
    // Call GPU search function
    search_on_gpu(storage, index.tree, SearchEF, M, index.max_elements_);
    
    cudaDeviceSynchronize();
    printf("GPU search completed!\n");
    
    return 0;
}