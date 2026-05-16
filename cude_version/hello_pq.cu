// GPU-accelerated iRangeGraph Search with PQ compression
// Using CUDA for greedy graph traversal with all SearchEF values
// PQ distances replace L2 for distance computation

#include <stdio.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <map>
#include <tuple>
#include <memory>
#include "iRG_search.h"
#include <curand_kernel.h>
#include "gpu_index.cuh"
#include "gpu_heap.cuh"
#include "gpu_visited.cuh"
#include "gpu_search_updated.cuh"  // Contains irange_search_kernel (normal)
#include "gpu_search_pq.cuh"       // Contains irange_search_kernel_pq (PQ version)
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>


const int query_K = 10;
int M_graph = 32;  // Example value, adjust as needed
int M_compression = 320;  // Example value, adjust as needed

#define THREADS_PER_QUERY_VAL 128
#define MAX_THREADS_PER_BLOCK 2000

using std::cout;
std::unordered_map<std::string, std::string> paths;


void init()
{
    paths["data_vector"] = "";
    paths["query_vector"] = "";
    paths["index"] = "";
    paths["result_saveprefix"] = "";
    paths["range_saveprefix"] = "";
    paths["groundtruth_saveprefix"] = "";
    paths["pq_model"] = "";
    paths["pq_codes"] = "";
}

// Load PQ model from Faiss
std::unique_ptr<faiss::ProductQuantizer> load_pq_model(const std::string& model_path) {
    printf("Loading PQ model from: %s\n", model_path.c_str());
    std::unique_ptr<faiss::ProductQuantizer> pq(faiss::read_ProductQuantizer(model_path.c_str()));
    if (!pq) {
        throw std::runtime_error("Failed to load PQ model from " + model_path);
    }
    printf("✓ PQ model loaded: d=%d, M=%d, nbits=%d, ksub=%d, dsub=%d, code_size=%d\n",
           pq->d, pq->M, pq->nbits, pq->ksub, pq->dsub, pq->code_size);
    return pq;
}

// Load PQ codes blob (using the definition from iRG_search.h)
PQCodesBlob load_pq_codes(const std::string& codes_path) {
    printf("Loading PQ codes from: %s\n", codes_path.c_str());
    std::ifstream in(codes_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open " + codes_path);
    }
    
    PQCodesBlob blob;
    in.read(reinterpret_cast<char*>(&blob.n), sizeof(int));
    in.read(reinterpret_cast<char*>(&blob.d), sizeof(int));
    in.read(reinterpret_cast<char*>(&blob.M), sizeof(int));
    in.read(reinterpret_cast<char*>(&blob.nbits), sizeof(int));
    in.read(reinterpret_cast<char*>(&blob.code_size), sizeof(int));
    
    if (!in || blob.n <= 0 || blob.d <= 0 || blob.M <= 0) {
        throw std::runtime_error("Invalid PQ code header in " + codes_path);
    }
    
    const size_t payload_size = (size_t)blob.n * blob.code_size;
    blob.codes.resize(payload_size);
    in.read(reinterpret_cast<char*>(blob.codes.data()), payload_size);
    
    if (!in) {
        throw std::runtime_error("Failed to read PQ codes from " + codes_path);
    }
    
    printf("✓ PQ codes loaded: n=%d, d=%d, M=%d, code_size=%d\n", 
           blob.n, blob.d, blob.M, blob.code_size);
    return blob;
}

void Generate(iRangeGraph::DataLoader &storage)
{
    storage.LoadData(paths["data_vector"]);
    iRangeGraph::QueryGenerator generator(storage.data_nb, storage.query_nb);
    generator.GenerateRange(paths["range_saveprefix"]);
    storage.LoadQueryRange(paths["range_saveprefix"]);
    generator.GenerateGroundtruth(paths["groundtruth_saveprefix"], storage);
}

void load_index_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    int dimension = index.storage->Dim;
    int data_points = index.max_elements_;
    size_t total_index_memory = (size_t)data_points * index.size_data_per_element_;
    
    // Calculate memory breakdown
    size_t embeddings_memory = (size_t)data_points * dimension * sizeof(float);
    size_t links_memory = total_index_memory - embeddings_memory;
    
    printf("\n========== INDEX MEMORY BREAKDOWN (DIAGNOSTIC) ==========\n");
    printf("Data points: %d\n", data_points);
    printf("Dimension: %d\n", dimension);
    printf("size_data_per_element: %zu bytes/point\n", index.size_data_per_element_);
    printf("size_links_per_element: %zu bytes/point\n", index.size_links_per_element_);
    printf("offsetData: %zu bytes (start offset for embeddings)\n", index.offsetData_);
    
    printf("\nTheoretical calculation (assuming layout):\n");
    printf("  IF layout is [adjacency lists][embeddings]:\n");
    printf("    Embeddings: %zu bytes (%.2f GB) = %d points × %d dims × 4 bytes\n", 
           embeddings_memory, embeddings_memory / 1e9, data_points, dimension);
    printf("    Adjacency lists: %zu bytes (%.2f GB)\n", links_memory, links_memory / 1e9);
    
    printf("\nActual memory layout (from data_memory_ structure):\n");
    printf("  offsetData=%zu suggests embeddings START at byte %zu\n", index.offsetData_, index.offsetData_);
    printf("  If offsetData > 0: embeddings are AFTER adjacency lists (adjlist comes first)\n");
    printf("  If offsetData = 0: embeddings are FIRST\n");
    
    // Sample first few bytes to understand layout
    printf("\nSampling first point (pid=0) memory layout:\n");
    char* base = (char*)index.data_memory_;
    printf("  Bytes 0-3 (first int of adjacency list): ");
    int first_val = *((int*)(base + 0));
    printf("%d\n", first_val);
    
    printf("  Bytes at offsetData=%zu (should be embedding start): ", index.offsetData_);
    float first_embedding = *((float*)(base + index.offsetData_));
    printf("%f\n", first_embedding);
    
    printf("\nPer-point memory structure:\n");
    printf("  size_links_per_element = %zu bytes\n", index.size_links_per_element_);
    printf("  size_data_per_element = %zu bytes\n", index.size_data_per_element_);
    printf("  Difference (padding/embeddings): %zu bytes\n", 
           index.size_data_per_element_ - index.size_links_per_element_);
    
    printf("\nTOTAL INDEX MEMORY: %zu bytes (%.2f GB)\n", total_index_memory, total_index_memory / 1e9);
    printf("=========================================\n\n");
    
    // Set metadata
    gpu_index.d_dim = dimension;
    gpu_index.d_size_data_per_element = index.size_data_per_element_;
    gpu_index.d_size_links_per_element = index.size_links_per_element_;
    gpu_index.d_offsetData = index.offsetData_;
    
    // Allocate GPU memory for entire index structure
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_data_memory, total_index_memory);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Copy entire index structure from CPU to GPU
    err = cudaMemcpy(gpu_index.d_data_memory, index.data_memory_, total_index_memory, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_data_memory);
        return;
    }
    printf("✓ Copied %.2f GB from CPU to GPU (index + embeddings + adjacency lists)\n", 
            total_index_memory / (1024.0*1024.0*1024.0));
}

void load_index_compact_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index, 
                               const std::string& index_path) {
    // Extract adjacency lists from CPU memory (data_memory_ is already loaded and padded)
    // This is for PQ mode: we copy ONLY the adjacency list portions, avoiding embeddings and padding
    
    printf("\n========== COMPACT INDEX LOADING (PQ MODE) ==========\n");
    printf("Extracting adjacency lists from CPU memory (compact transfer to GPU)\n");
    
    int data_points = index.max_elements_;
    size_t size_links_per_element = index.size_links_per_element_;
    size_t size_data_per_element = index.size_data_per_element_;  // Includes padding
    
    printf("Memory calculation (compact, no padding):\n");
    printf("  Data points: %d\n", data_points);
    printf("  Adjacency per point (actual): %zu bytes\n", size_links_per_element);
    printf("  Total padded size per point: %zu bytes (padding: %zu bytes)\n", 
           size_data_per_element, size_data_per_element - size_links_per_element);
    printf("  TOTAL adjacency lists only: %zu bytes (%.2f MB)\n", 
           (size_t)data_points * size_links_per_element, 
           ((size_t)data_points * size_links_per_element) / (1024.0*1024.0));
    printf("  OLD total with padding: %zu bytes (%.2f MB)\n",
           (size_t)data_points * size_data_per_element,
           ((size_t)data_points * size_data_per_element) / (1024.0*1024.0));
    
    size_t total_compact_memory = (size_t)data_points * size_links_per_element;
    
    // Allocate GPU memory for d_data_memory but only with adjacency list size (no padding!)
    // This way kernels can still access through d_data_memory pointer
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_data_memory, total_compact_memory);
    if (err != cudaSuccess) {
        printf("ERROR: CudaMalloc failed for compact adjacency lists: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Also keep track for cleanup later
    gpu_index.d_adjacency_lists = gpu_index.d_data_memory;
    gpu_index.d_adjacency_lists_size = total_compact_memory;
    
    // Copy adjacency lists sequentially from CPU memory
    // Note: Each point starts at data_memory_ + i * size_data_per_element
    // Adjacency list is the first size_links_per_element bytes of each point
    printf("Copying adjacency lists from CPU to GPU (compacting out padding)...\n");
    
    char* host_buffer = new char[total_compact_memory];
    if (!host_buffer) {
        printf("ERROR: Failed to allocate host buffer\n");
        cudaFree(gpu_index.d_data_memory);
        return;
    }
    
    // Extract adjacency lists without padding
    for (int i = 0; i < data_points; i++) {
        char* src = (char*)index.data_memory_ + i * size_data_per_element;
        char* dst = host_buffer + i * size_links_per_element;  // Calculate offset directly
        memcpy(dst, src, size_links_per_element);
    }
    
    // Transfer to GPU in one go
    err = cudaMemcpy(gpu_index.d_data_memory, host_buffer, total_compact_memory, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERROR: CudaMemcpy failed for adjacency lists: %s\n", cudaGetErrorString(err));
        delete[] host_buffer;
        cudaFree(gpu_index.d_data_memory);
        return;
    }
    
    delete[] host_buffer;
    
    // Set metadata - CRITICAL: update size_data_per_element to compact size for kernel access
    gpu_index.d_dim = index.storage->Dim;
    gpu_index.d_size_data_per_element = size_links_per_element;  // COMPACT MODE: only adjacency lists
    gpu_index.d_size_links_per_element = index.size_links_per_element_;
    gpu_index.d_offsetData = 0;  // No embeddings in compact mode
    
    printf("✓ Copied %.2f MB compact adjacency lists to d_data_memory (NO embeddings, NO padding)\n", 
           total_compact_memory / (1024.0*1024.0));
    printf("✓ Updated d_size_data_per_element to %zu (compact) for kernel access\n", size_links_per_element);
    
    printf("\n[GPU TRANSFER] Adjacency Lists (Compact):\n");
    printf("  - Size per point: %zu bytes (links only)\n", size_links_per_element);
    printf("  - Total points: %d\n", data_points);
    printf("  - Total transferred: %.2f MB (%.4f GB)\n", 
           total_compact_memory / (1024.0*1024.0), total_compact_memory / (1024.0*1024.0*1024.0));
    printf("  - Reduction vs CPU padded: %.1f%%\n", 
           100.0 * (1.0 - (double)total_compact_memory / ((size_t)data_points * size_data_per_element)));
    printf("===================================================\n\n");
}

void load_segment_tree_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    iRangeGraph::SegmentTree *tree = index.tree;
    std::vector<GPUNode> gpu_nodes = tree->FlattenGPUTree();
    
    size_t mem_to_allocate_to_gpu = gpu_nodes.size() * sizeof(GPUNode);

    // Initialize segment tree structure
    gpu_index.d_segment_tree.num_nodes = gpu_nodes.size();
    gpu_index.d_segment_tree.max_depth = tree->max_depth;
    gpu_index.d_segment_tree.root = gpu_nodes[0];
    
    // Allocate GPU memory for nodes
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_segment_tree.d_nodes, mem_to_allocate_to_gpu);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for segment tree: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy nodes to GPU
    err = cudaMemcpy(gpu_index.d_segment_tree.d_nodes, gpu_nodes.data(), mem_to_allocate_to_gpu, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for segment tree: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_segment_tree.d_nodes);
        return;
    }
    
    printf("✓ Copied %zu nodes (%.2f KB) to GPU\n", 
           gpu_index.d_segment_tree.num_nodes, mem_to_allocate_to_gpu / 1024.0);
    
    // Run test kernel to verify tree structure
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Test kernel error: %s\n", cudaGetErrorString(err));
    }
}


void load_queries_to_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    int query_nb = index.storage->query_nb;
    int dim = index.storage->Dim;
    
    // Allocate GPU memory for query vectors
    size_t query_vectors_size = (size_t)query_nb * dim * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_query_vectors, query_vectors_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for query vectors: %s\n", cudaGetErrorString(err));
        return;
    }
    
    float *flatten_queries = new float[query_nb * dim];
    for (int i = 0; i < query_nb; i++) {
        for (int d = 0; d < dim; d++) {
            flatten_queries[i * dim + d] = index.storage->query_points[i][d];
        }
    }

    // Copy query vectors to GPU
    err = cudaMemcpy(gpu_index.d_query_vectors, flatten_queries, query_vectors_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for query vectors: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_query_vectors);
        delete[] flatten_queries;
        return;
    }
    
    // Allocate GPU memory for query ranges
    // query_range is a map with suffix keys
    std::vector<int> suffix_keys;
    for (auto range : index.storage->query_range) {
        suffix_keys.push_back(range.first);
    }

    size_t query_ranges_size = (size_t)query_nb * 2 * sizeof(int) * suffix_keys.size();
    err = cudaMalloc((void**)&gpu_index.d_query_range, query_ranges_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for query ranges: %s\n", cudaGetErrorString(err));
        delete[] flatten_queries;
        return;
    }
    
    int *flatten_ranges = new int[query_nb * 2 * suffix_keys.size()];
    for (int i = 0; i < query_nb; i++) {
        for (size_t s = 0; s < suffix_keys.size(); s++) {
            int suffix = suffix_keys[s];
            flatten_ranges[i * 2 * suffix_keys.size() + s * 2] = index.storage->query_range[suffix][i].first;
            flatten_ranges[i * 2 * suffix_keys.size() + s * 2 + 1] = index.storage->query_range[suffix][i].second;
        }
    }

    // Copy query ranges to GPU
    err = cudaMemcpy(gpu_index.d_query_range, flatten_ranges, query_ranges_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for query ranges: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_query_range);
        delete[] flatten_queries;
        delete[] flatten_ranges;
        return;
    }
    
    delete[] flatten_queries;
    delete[] flatten_ranges;
    
    size_t query_vectors_transferred = (size_t)query_nb * dim * sizeof(float);
    size_t query_ranges_transferred = (size_t)query_nb * 2 * sizeof(int) * suffix_keys.size();
    
    printf("✓ Copied %d queries (%d suffixes) and ranges to GPU\n", query_nb, (int)suffix_keys.size());
    
    printf("\n[GPU TRANSFER] Query Vectors & Ranges:\n");
    printf("  - Query vectors: %d vectors × %d dims × 4 bytes = %.2f MB\n", 
           query_nb, dim, query_vectors_transferred / (1024.0*1024.0));
    printf("  - Query ranges: %d queries × %zu suffixes × 2 ints = %.2f MB\n",
           query_nb, suffix_keys.size(), query_ranges_transferred / (1024.0*1024.0));
    printf("  - Total query data: %.2f MB\n", 
           (query_vectors_transferred + query_ranges_transferred) / (1024.0*1024.0));
}

int* make_result_buffer_on_gpu(iRangeGraph::iRangeGraph_Search<float> &index, GPUIndex &gpu_index) {
    int query_nb = index.storage->query_nb;
    size_t result_buffer_size = (size_t)query_nb * query_K * sizeof(int);

    cudaError_t err = cudaMalloc((void**)&gpu_index.d_results, result_buffer_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for result buffer: %s\n", cudaGetErrorString(err));        
        return NULL;
    }

    return gpu_index.d_results;

}

void load_pq_model_to_gpu(GPUIndex &gpu_index, faiss::ProductQuantizer* pq_model) {
    gpu_index.pq_dsub = pq_model->dsub;
    gpu_index.pq_ksub = pq_model->ksub;
    gpu_index.pq_M = pq_model->M;
    gpu_index.pq_nbits = pq_model->nbits;
    gpu_index.pq_code_size = pq_model->code_size;
    gpu_index.use_pq = true;

    size_t centroids_size = (size_t)pq_model->M * pq_model->ksub * pq_model->dsub * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_centroids, centroids_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for PQ centroids: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(gpu_index.d_centroids, pq_model->centroids.data(), centroids_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for PQ centroids: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_centroids);
        return;
    }
    
    printf("✓ Loaded PQ model to GPU: M=%d, nbits=%d, dsub=%d, ksub=%d\n", 
           gpu_index.pq_M, gpu_index.pq_nbits, gpu_index.pq_dsub, gpu_index.pq_ksub);
    
    size_t centroids_memory = centroids_size;
    printf("\n[GPU TRANSFER] PQ Model Centroids:\n");
    printf("  - Subspaces (M): %d\n", pq_model->M);
    printf("  - Centroids per subspace (ksub): %d\n", pq_model->ksub);
    printf("  - Dimension per subspace (dsub): %d\n", pq_model->dsub);
    printf("  - Total centroids: %ld (M × ksub × dsub = %d × %d × %d)\n", 
           centroids_memory / sizeof(float), pq_model->M, pq_model->ksub, pq_model->dsub);
    printf("  - Total transferred: %.2f MB\n", centroids_memory / (1024.0*1024.0));
    printf("  - Format: float32\n");
}

void load_pq_codes_to_gpu(GPUIndex &gpu_index, const std::vector<uint8_t>& pq_codes,
                          int n_vectors, int M, int nbits, int code_size, int ksub, int dsub) {
    gpu_index.pq_codes_cpu = pq_codes;  // Keep a copy on CPU for reference

    gpu_index.gpu_codes_size = (size_t)pq_codes.size() * sizeof(uint8_t);
    
    // Calculate PQ centroids memory
    size_t centroids_size = (size_t)M * ksub * dsub * sizeof(float);
    
    printf("\n========== PQ MEMORY BREAKDOWN ==========\n");
    printf("PQ Model parameters: n=%d, M=%d, nbits=%d, ksub=%d, dsub=%d\n",
           n_vectors, M, nbits, ksub, dsub);
    printf("\nPQ-specific allocations:\n");
    printf("  PQ codes: %zu bytes (%.2f MB) = %d vectors × %d bytes/code\n", 
           gpu_index.gpu_codes_size, gpu_index.gpu_codes_size / (1024.0*1024.0), 
           n_vectors, code_size);
    printf("  PQ centroids: %zu bytes (%.2f MB) = %d subspaces × %d centroids × %d dims × 4 bytes\n",
           centroids_size, centroids_size / (1024.0*1024.0), M, ksub, dsub);
    printf("  TOTAL PQ: %.2f MB\n", (gpu_index.gpu_codes_size + centroids_size) / (1024.0*1024.0));
    printf("=========================================\n\n");
    
    cudaError_t err = cudaMalloc((void**)&gpu_index.d_compressed_codes, gpu_index.gpu_codes_size);
    if (err != cudaSuccess) {
        printf("CudaMalloc failed for PQ codes: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(gpu_index.d_compressed_codes, pq_codes.data(), gpu_index.gpu_codes_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CudaMemcpy failed for PQ codes: %s\n", cudaGetErrorString(err));
        cudaFree(gpu_index.d_compressed_codes);
        return;
    }
    
    printf("✓ Loaded %zu PQ codes to GPU (%.2f MB)\n", 
           pq_codes.size(), gpu_index.gpu_codes_size / (1024.0*1024.0));
    
    printf("\n[GPU TRANSFER] PQ Codes (Compressed Vectors):\n");
    printf("  - Total vectors: %d\n", n_vectors);
    printf("  - Code size per vector: %d bytes (M=%d subspaces, %d bits each)\n",
           code_size, M, nbits);
    printf("  - Total codes: %zu bytes (%.2f MB)\n", 
           gpu_index.gpu_codes_size, gpu_index.gpu_codes_size / (1024.0*1024.0));
    printf("  - Format: uint8_t (byte-aligned codes)\n");
    printf("  - Compression ratio vs full vectors: %.1f%% of full float32\n",
           100.0 * gpu_index.gpu_codes_size / (n_vectors * 96 * sizeof(float)));
}

void write_results_to_csv(const std::string& saveprefix, int suffix, 
                          const std::vector<std::tuple<int, float, float, float, float>>& results) {
    printf("\n========================================\n");
    printf("Writing results for suffix %d to disk...\n", suffix);
    printf("========================================\n");
    
    std::string savepath = saveprefix + std::to_string(suffix) + "_gpu.csv";
    CheckPath(savepath);
    std::ofstream outfile(savepath);
    
    if (outfile.is_open()) {
        // Write header
        outfile << "SearchEF,Recall,QPS,DCO,HOP\n";
        
        // Write all results for this suffix
        for (const auto& result : results) {
            int ef = std::get<0>(result);
            float recall = std::get<1>(result);
            float qps = std::get<2>(result);
            float dco = std::get<3>(result);
            float hop = std::get<4>(result);
            
            outfile << ef << "," << recall << "," << qps << "," << dco << "," << hop << "\n";
        }
        
        outfile.close();
        printf("✓ Saved %zu results to %s\n", results.size(), savepath.c_str());
    } else {
        printf("✗ Failed to open %s\n", savepath.c_str());
    }
}

void search_on_gpu(iRangeGraph::iRangeGraph_Search<float> &index, std::vector<int> SearchEF, std::string saveprefix, 
                   faiss::ProductQuantizer* pq_model, PQCodesBlob& pq_blob) {
    // Validate all SearchEF values are within supported range
    for (int ef : SearchEF) {
        if (ef > MAX_SEARCH_EF) {
            printf("ERROR: SearchEF=%d exceeds MAX_SEARCH_EF=%d\n", ef, MAX_SEARCH_EF);
            printf("Please increase MAX_SEARCH_EF in gpu_search_updated.cuh or reduce SearchEF\n");
            return;
        }
    }

    GPUIndex gpu_index;

    

    printf("\n>>> PQ MODE DETECTED: Loading COMPACT adjacency lists only\n");
    load_index_compact_to_gpu(index, gpu_index, paths["index"]);
   
    // Load segment tree to GPU
    load_segment_tree_to_gpu(index, gpu_index);
    load_queries_to_gpu(index, gpu_index);
    make_result_buffer_on_gpu(index, gpu_index);

    load_pq_model_to_gpu(gpu_index, pq_model);
    load_pq_codes_to_gpu(gpu_index, pq_blob.codes, pq_blob.n, pq_blob.M, pq_blob.nbits, pq_blob.code_size, pq_model->ksub, pq_model->dsub);
    
    // ===== PRINT GPU MEMORY SUMMARY =====
    size_t adjacency_lists_size = (size_t)index.max_elements_ * index.size_links_per_element_;
    size_t pq_codes_size = pq_blob.codes.size();
    size_t pq_centroids_size = (size_t)pq_blob.M * pq_model->ksub * pq_model->dsub * sizeof(float);
    int query_nb = index.storage->query_nb;
    int dim = index.storage->Dim;
    size_t query_vectors_size = (size_t)query_nb * dim * sizeof(float);
    
    // Count unique suffixes for ranges
    int num_suffixes = index.storage->query_range.size();
    size_t query_ranges_size = (size_t)query_nb * 2 * sizeof(int) * num_suffixes;
    
    size_t total_gpu_memory = adjacency_lists_size + pq_codes_size + pq_centroids_size + 
                              query_vectors_size + query_ranges_size;
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║         GPU MEMORY ALLOCATION SUMMARY (PQ MODE)           ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║ COMPONENT                          │ SIZE (MB)  │ SIZE (GB) ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║ Adjacency Lists (Compact)          │ %10.2f │ %9.4f ║\n",
           adjacency_lists_size / (1024.0*1024.0), adjacency_lists_size / (1024.0*1024.0*1024.0));
    printf("║ PQ Compressed Codes (uint8)        │ %10.2f │ %9.4f ║\n",
           pq_codes_size / (1024.0*1024.0), pq_codes_size / (1024.0*1024.0*1024.0));
    printf("║ PQ Centroids (float32)             │ %10.2f │ %9.4f ║\n",
           pq_centroids_size / (1024.0*1024.0), pq_centroids_size / (1024.0*1024.0*1024.0));
    printf("║ Query Vectors (float32)            │ %10.2f │ %9.4f ║\n",
           query_vectors_size / (1024.0*1024.0), query_vectors_size / (1024.0*1024.0*1024.0));
    printf("║ Query Ranges (int32)               │ %10.2f │ %9.4f ║\n",
           query_ranges_size / (1024.0*1024.0), query_ranges_size / (1024.0*1024.0*1024.0));
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║ TOTAL GPU MEMORY USED              │ %10.2f │ %9.4f ║\n",
           total_gpu_memory / (1024.0*1024.0), total_gpu_memory / (1024.0*1024.0*1024.0));
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\nKey Facts:\n");
    printf("  ✓ NO full embeddings loaded (replaced by PQ codes)\n");
    printf("  ✓ NO padding in adjacency lists (compact layout)\n");
    printf("  ✓ Adjacency lists: %d bytes/point (no embeddings, no padding)\n", 
           (int)index.size_links_per_element_);
    printf("  ✓ PQ Code size: %d bytes/vector (vs %d bytes for full float32)\n",
           pq_blob.code_size, dim * (int)sizeof(float));
    printf("  ✓ Query vectors: %d × %d dims\n", query_nb, dim);
    printf("  ✓ Distance calculation: Using PQDistance() with compressed codes\n");
    printf("\n");
    
    
   
    
    // Structure to store results for current suffix before writing
    // Vector of (SearchEF, Recall, QPS, DCO, HOP) for current suffix
    std::vector<std::tuple<int, float, float, float, float>> current_suffix_results;
    
    // Allocate ADC distance table buffer: one flat table per query.
    // Each table is pq_M * pq_ksub floats. Allocated once, reused across all ef values and suffixes.
    {
        int qnb = index.storage->query_nb;
        size_t table_floats = (size_t)gpu_index.pq_M * gpu_index.pq_ksub;
        size_t table_bytes  = (size_t)qnb * table_floats * sizeof(float);
        cudaError_t err = cudaMalloc(&gpu_index.d_dist_tables, table_bytes);
        if (err != cudaSuccess) {
            printf("ERROR: Failed to allocate ADC distance tables: %s\n", cudaGetErrorString(err));
            return;
        }
        printf("✓ ADC distance tables allocated: %d queries × %zu floats = %.1f MB\n",
               qnb, table_floats, table_bytes / (1024.0 * 1024.0));
    }

    // Iterate over all suffixes in storage->query_range (same as CPU version)
    size_t suffix_idx = 0;
    for (auto range : index.storage->query_range) {
        int suffix = range.first;
        printf("\n========================================\n");
        printf("Processing suffix %d (%zu/%zu)\n", suffix, suffix_idx + 1, index.storage->query_range.size());
        printf("========================================\n");
        
        // Clear results for this suffix
        current_suffix_results.clear();

        for (int ef : SearchEF) {
            int query_nb = index.storage->query_nb;
            int max_elements = index.max_elements_;
            int dim = index.storage->Dim;
            
            GPUVisitedArray visited;
            cudaError_t err = initGPUVisitedArray(visited, query_nb, max_elements);
            if (err != cudaSuccess) {
                printf("Failed to initialize visited array: %s\n", cudaGetErrorString(err));
                return;
            }
            
            // Allocate metrics arrays on GPU
            int* d_hops;
            int* d_dist_comps;
            cudaMalloc(&d_hops, query_nb * sizeof(int));
            cudaMalloc(&d_dist_comps, query_nb * sizeof(int));
            
            // Initialize to zero (CRITICAL - otherwise contains garbage values!)
            cudaMemset(d_hops, 0, query_nb * sizeof(int));
            cudaMemset(d_dist_comps, 0, query_nb * sizeof(int));
            
            int threads_per_block = THREADS_PER_QUERY_VAL;   // 128: 1 query per block
            int threads_per_query = THREADS_PER_QUERY_VAL;   // 128 threads collaborate on each query
            int queries_per_block = threads_per_block / threads_per_query;  // = 1
            int num_blocks = (query_nb + queries_per_block - 1) / queries_per_block;
            

            
            // Reset visited counters for new search
            unsigned short* temp_curV = new unsigned short[query_nb];
            for (int i = 0; i < query_nb; i++) temp_curV[i] = 1;
            cudaMemcpy(visited.d_curV, temp_curV, query_nb * sizeof(unsigned short), cudaMemcpyHostToDevice);
            cudaMemset(visited.d_mass, 0, (size_t)query_nb * max_elements * sizeof(unsigned short));
            delete[] temp_curV;
            
            // CRITICAL: Clear output buffers before each SearchEF iteration
            cudaMemset(d_hops, 0, query_nb * sizeof(int));
            cudaMemset(d_dist_comps, 0, query_nb * sizeof(int));
            
            // Launch appropriate kernel (PQ or normal) - no branching inside kernel
            unsigned long long kernel_seed = std::chrono::system_clock::now().time_since_epoch().count();
            
            printf("    suffix %d  ef=%d  launching kernel...\n", suffix, ef);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            
            irange_search_kernel_pq<<<num_blocks, threads_per_block>>>(
                gpu_index, visited, query_nb, ef, query_K, dim, suffix_idx, d_hops, d_dist_comps,
                index.size_links_per_layer_, kernel_seed
            );
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            float searchtime = milliseconds / 1000.0f;  // Convert to seconds
            
            cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Search kernel error: %s\n", cudaGetErrorString(err));
                continue;
            }
            
            // Copy results back to CPU
            int* cpu_results = new int[query_nb * query_K];
            int* cpu_hops = new int[query_nb];
            int* cpu_dist_comps = new int[query_nb];
            
            cudaMemcpy(cpu_results, gpu_index.d_results, query_nb * query_K * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_hops, d_hops, query_nb * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_dist_comps, d_dist_comps, query_nb * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Compute recall if ground truth is available
            int tp = 0;
            if (index.storage->groundtruth.count(suffix)) {
                auto &gt = index.storage->groundtruth[suffix];
                for (int i = 0; i < query_nb; i++) {
                    for (int k = 0; k < query_K; k++) {
                        int result_id = cpu_results[i * query_K + k];
                        if (result_id != -1) {
                            if (std::find(gt[i].begin(), gt[i].end(), result_id) != gt[i].end()) {
                                tp++;
                            }
                        }
                    }
                }
            }
            
            // Calculate metrics
            float recall = (index.storage->groundtruth.count(suffix)) ? 
                            (1.0f * tp / query_nb / query_K) : 0.0f;
            float qps = query_nb / searchtime;
            
            long long total_hops = 0;
            long long total_dist_comps = 0;
            for (int i = 0; i < query_nb; i++) {
                total_hops += cpu_hops[i];
                total_dist_comps += cpu_dist_comps[i];
            }
            float avg_hops = (float)total_hops / query_nb;
            float avg_dco = (float)total_dist_comps / query_nb;

            printf("    suffix %d  ef=%d  recall=%.4f  QPS=%.1f  DCO=%.1f  HOP=%.1f  time=%.3fs\n",
                   suffix, ef, recall, qps, avg_dco, avg_hops, searchtime);
            
            // Store results for this suffix
            current_suffix_results.push_back(std::make_tuple(ef, recall, qps, avg_dco, avg_hops));
            
            delete[] cpu_results;
            delete[] cpu_hops;
            delete[] cpu_dist_comps;
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            
            // Clean up
            cudaFree(d_hops);
            cudaFree(d_dist_comps);
            freeGPUVisitedArray(visited);
            
        }

        // Write results to CSV file for this suffix
        write_results_to_csv(saveprefix, suffix, current_suffix_results);

        suffix_idx++;  // Increment for next suffix

    }
    
    
    // Free ADC distance table buffer
    if (gpu_index.d_dist_tables) {
        cudaFree(gpu_index.d_dist_tables);
        gpu_index.d_dist_tables = nullptr;
    }

    printf("\n========================================\n");
    printf("All suffixes processed and results written!\n");
    printf("========================================\n");


    // Initialize visited array for all queries
    int size_of_node = sizeof(iRangeGraph::TreeNode);
    printf("\nSize of TreeNode: %d bytes\n", size_of_node);
    int size_of_tree = sizeof(iRangeGraph::SegmentTree);
    printf("Size of SegmentTree: %d bytes\n", size_of_tree);
    
    // Clean up GPU memory
    if (gpu_index.d_data_memory) cudaFree(gpu_index.d_data_memory);
    if (gpu_index.d_adjacency_lists) cudaFree(gpu_index.d_adjacency_lists);
    if (gpu_index.d_segment_tree.d_nodes) cudaFree(gpu_index.d_segment_tree.d_nodes);
    if (gpu_index.d_query_vectors) cudaFree(gpu_index.d_query_vectors);
    if (gpu_index.d_query_range) cudaFree(gpu_index.d_query_range);
    if (gpu_index.d_results) cudaFree(gpu_index.d_results);
    if (gpu_index.d_compressed_codes) cudaFree(gpu_index.d_compressed_codes);
    if (gpu_index.d_centroids) cudaFree(gpu_index.d_centroids);
    
    printf("✓ GPU memory cleaned up\n");
}



int main(int argc, char **argv) {

    setvbuf(stdout, NULL, _IOLBF, 0);  // line-buffered stdout: live progress in SLURM logs

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path_comp")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--index_path")
            paths["index"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];
        if (arg == "--range_saveprefix")
            paths["range_saveprefix"] = argv[i + 1];
        if (arg == "--groundtruth_saveprefix")
            paths["groundtruth_saveprefix"] = argv[i + 1];
        if (arg == "--pq_model_out")
            paths["pq_model"] = argv[i + 1];
        if (arg == "--pq_codes_out")
            paths["pq_codes"] = argv[i + 1];
        if (arg == "--M_compression_spaces")
            M_compression = std::stoi(argv[i + 1]);
        if (arg == "--graph_M")
            M_graph = std::stoi(argv[i + 1]);  // Can also override with graph_M
    }

    if (argc < 15) {
        std::cerr << "Usage: " << argv[0] << std::endl
                  << "  --data_path_comp <path>" << std::endl
                  << "  --query_path <path>" << std::endl
                  << "  --index_path <path>" << std::endl
                  << "  --result_saveprefix <path>" << std::endl
                  << "  --range_saveprefix <path>" << std::endl
                  << "  --groundtruth_saveprefix <path>" << std::endl
                  << "  --pq_model_out <path>" << std::endl
                  << "  --pq_codes_out <path>" << std::endl
                  << "  --M_compression_spaces <int>" << std::endl
                  << "  [--graph_M <int>] (default: 32 - edges per node in graph)" << std::endl;
        return 1;
    }
    
    if (paths["data_vector"].empty() || paths["query_vector"].empty() || paths["index"].empty() ||
        paths["result_saveprefix"].empty() || paths["range_saveprefix"].empty() || 
        paths["groundtruth_saveprefix"].empty() || paths["pq_model"].empty() || paths["pq_codes"].empty()) {
        std::cerr << "Error: All path arguments are required" << std::endl;
        return 1;
    }
    

    iRangeGraph::DataLoader storage;
    storage.query_K = query_K;
    std::cout << "Loading queries..." << std::endl;
    storage.LoadQuery(paths["query_vector"]);
    // Generate(storage);
    std::cout << "Loading query ranges..." << std::endl;
    storage.LoadQueryRange(paths["range_saveprefix"]);
    std::cout << "Loading ground truth..." << std::endl;
    storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);

    std::cout << "Loading index..." << std::endl;
    iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M_graph);

    int data_memory_size = index.max_elements_ * index.size_data_per_element_;  
    printf("the size of data_memory_ loaded in index: %zu bytes\n", data_memory_size);
    printf("the size of data per element: %zu bytes\n", index.size_data_per_element_);
    printf("the size of links per element: %zu bytes\n", index.size_links_per_element_);
    printf("the offset for data (embeddings) in each element: %zu bytes\n", index.offsetData_);

    int memory_for_links = index.max_elements_ * index.size_links_per_element_;

    printf("theoretical memory for adjacency lists (links only): %zu bytes (%.2f GB)\n", 
           memory_for_links, memory_for_links / 1e9);

    
    // Create GPUIndex - it will own all GPU memory
    printf("\n========================================\n");
    printf("Creating GPUIndex structure...\n");
    printf("========================================\n");    

    
    // SearchEF values to test (from largest to smallest for better performance)
    std::vector<int> SearchEF_values = {1700, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10};
    
    // Load PQ codes to set use_pq_ flag BEFORE search_on_gpu, so compact loader is used
    std::unique_ptr<faiss::ProductQuantizer> pq_model_for_flag = load_pq_model(paths["pq_model"]);
    PQCodesBlob pq_blob_for_flag = load_pq_codes(paths["pq_codes"]);
    index.load_pq_codes(pq_blob_for_flag.codes, pq_blob_for_flag.code_size, 
                        pq_blob_for_flag.M, pq_blob_for_flag.nbits, pq_model_for_flag.get());
    
    search_on_gpu(index, SearchEF_values, paths["result_saveprefix"], pq_model_for_flag.get(), pq_blob_for_flag);
    
    
    printf("\n========================================\n");
    printf("Cleanup complete\n");
    printf("========================================\n");
    
    return 0;
}
