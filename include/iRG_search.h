#pragma once

#include <vector>
#include <sys/resource.h>
#include "utils.h"
#include "searcher.hpp"
#include "memory.hpp"
#include <bitset>
#include <cuda_runtime.h>
#include <omp.h>
#include <limits>
#include <faiss/impl/ProductQuantizer.h>

// PQ Codes Blob structure
struct PQCodesBlob {
    int32_t n = 0;
    int32_t d = 0;
    int32_t M = 0;
    int32_t nbits = 0;
    int32_t code_size = 0;
    std::vector<uint8_t> codes;
};

namespace iRangeGraph
{
    template <typename dist_t>
    class iRangeGraph_Search
    {
    public:
        DataLoader *storage;
        SegmentTree *tree;
        size_t max_elements_{0};
        size_t dim_{0};
        size_t M_out{0};
        size_t ef_construction{0};

        size_t size_data_per_element_{0};
        size_t size_links_per_element_{0};
        size_t data_size_{0};

        size_t size_links_per_layer_{0};
        size_t offsetData_{0};

        char *data_memory_{nullptr};

        hnswlib::L2Space *space;
        hnswlib::DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_{nullptr};

        size_t metric_distance_computations{0};
        size_t metric_hops{0};

        int prefetch_lines{0};

        // PQ compressed codes members
        std::vector<uint8_t> compressed_codes_;
        int pq_code_size_{0};  // Size of each compressed code
        int pq_M_{0};          // Number of subquantizers
        int pq_nbits_{0};      // Bits per centroid
        bool use_pq_{false};   // Flag to use PQ codes
        faiss::ProductQuantizer* pq_model_{nullptr};  // Pointer to PQ codebook

        iRangeGraph_Search(std::string vectorfilename, std::string edgefilename, DataLoader *store, int M) : storage(store)
        {
            std::ifstream vectorfile(vectorfilename, std::ios::in | std::ios::binary);
            if (!vectorfile.is_open())
                throw Exception("cannot open " + vectorfilename);
            std::ifstream edgefile(edgefilename, std::ios::in | std::ios::binary);
            if (!edgefile.is_open())
                throw Exception("cannot open " + edgefilename);

            vectorfile.read((char *)&max_elements_, sizeof(int));
            vectorfile.read((char *)&dim_, sizeof(int));

            tree = new SegmentTree(max_elements_);
            tree->BuildTree(tree->root);

            space = new hnswlib::L2Space(dim_);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            M_out = M;

            data_size_ = (dim_ + 7) / 8 * 8 * sizeof(float);
            size_links_per_layer_ = M_out * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_per_element_ = (size_links_per_layer_ * (tree->max_depth + 1) + 31) / 32 * 32;
            size_data_per_element_ = size_links_per_element_ + data_size_;
            offsetData_ = size_links_per_element_;
            prefetch_lines = data_size_ >> 4;

            data_memory_ = (char *)memory::align_mm<1 << 21>(max_elements_ * size_data_per_element_);
            if (data_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            for (int pid = 0; pid < max_elements_; pid++)
            {
                for (int layer = 0; layer <= tree->max_depth; layer++)
                {
                    linklistsizeint *data = get_linklist(pid, layer);
                    edgefile.read((char *)data, sizeof(tableint));
                    int size = getListCount(data);
                    if (size > M_out)
                        throw Exception("real linklist size is bigger than defined M_out");
                    for (int i = 0; i < size; i++)
                    {
                        char *current_neighbor_ = (char *)(data + 1 + i);
                        edgefile.read(current_neighbor_, sizeof(tableint));
                    }
                }

                char *data = getDataByInternalId(pid);
                // vectorfile.read(data, data_size_);
                vectorfile.read(data, dim_ * sizeof(float));
            }

            edgefile.close();
            vectorfile.close();
            std::cout << "load index finished ..." << std::endl;
        }

        ~iRangeGraph_Search()
        {
            free(data_memory_);
            data_memory_ = nullptr;
        }

        // Load compressed PQ codes from blob
        void load_pq_codes(const std::vector<uint8_t>& codes, int code_size, int M, int nbits,
                           faiss::ProductQuantizer* pq_model = nullptr)
        {
            compressed_codes_ = codes;
            pq_code_size_ = code_size;
            pq_M_ = M;
            pq_nbits_ = nbits;
            pq_model_ = pq_model;
            use_pq_ = true;
        }

        // Helper: Extract nbits bits from byte array at given bit offset
        inline uint32_t extract_centroid_idx(const uint8_t* data, int subspace_idx, int nbits) const
        {
            int bit_offset = subspace_idx * nbits;
            uint32_t result = 0;
            for (int i = 0; i < nbits; ++i) {
                int byte_idx = (bit_offset + i) / 8;
                int bit_idx = (bit_offset + i) % 8;
                uint8_t bit = (data[byte_idx] >> bit_idx) & 1;
                result |= (bit << i);
            }
            return result;
        }

        // Compute distance between raw query vector and compressed DB vector using PQ codebook
        // This iterates through each subspace, finds the centroid for that subspace from the code,
        // and computes L2 distance between query subvector and centroid subvector
        float compute_pq_distance(const float* query_vector, int data_id) const
        {
            if (!pq_model_ || data_id < 0 || data_id >= max_elements_)
                return std::numeric_limits<float>::max();
            
            float distance = 0.0f;
            const uint8_t* data_code = compressed_codes_.data() + data_id * pq_code_size_;
            
            // Iterate through each subspace
            for (int m = 0; m < pq_M_; ++m) {
                // Extract centroid index for this subspace (handles nbits != 8)
                uint32_t centroid_idx = extract_centroid_idx(data_code, m, pq_nbits_);
                
                // Get pointers to query subvector and centroid subvector
                const float* query_m = query_vector + m * pq_model_->dsub;
                const float* centroid_m = pq_model_->get_centroids(m, 0) + centroid_idx * pq_model_->dsub;
                
                // Compute L2 distance in this subspace
                for (size_t d = 0; d < pq_model_->dsub; ++d) {
                    float diff = query_m[d] - centroid_m[d];
                    distance += diff * diff;
                }
            }
            
            return distance;
        }

        __device__ float L2Distance(const float *a, const float *b, size_t dim);

        inline char *getDataByInternalId(tableint internal_id) const
        {
            return (data_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        linklistsizeint *get_linklist(tableint internal_id, int layer) const
        {
            return (linklistsizeint *)(data_memory_ + internal_id * size_data_per_element_ + layer * size_links_per_layer_);
        }

        int getListCount(linklistsizeint *ptr) const
        {
            return *((int *)ptr);
        }

        int GetOverLap(int l, int r, int ql, int qr)
        {
            int L = std::max(l, ql);
            int R = std::min(r, qr);
            return R - L + 1;
        }

        std::vector<tableint> SelectEdge(int pid, int ql, int qr, int edge_limit, searcher::Bitset<uint64_t> &visited_set)
        {
            TreeNode *cur_node = nullptr, *nxt_node = tree->root;
            std::vector<tableint> selected_edges;
            selected_edges.reserve(edge_limit);
            do
            {
                cur_node = nxt_node;
                bool contain = false;
                do
                {
                    contain = false;
                    if (cur_node->childs.size() == 0)
                        nxt_node = nullptr;
                    else
                    {
                        for (int i = 0; i < cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                                break;
                            }
                        }
                        if (GetOverLap(cur_node->lbound, cur_node->rbound, ql, qr) == GetOverLap(nxt_node->lbound, nxt_node->rbound, ql, qr))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);

                int *data = (int *)get_linklist(pid, cur_node->depth);
                size_t size = getListCount((linklistsizeint *)data);

                for (size_t j = 1; j <= size; ++j)
                {
                    int neighborId = *(data + j);
                    if (neighborId < ql || neighborId > qr)
                        continue;
                    // if (visitedpool[neighborId] == visited_tag)
                    //     continue;
                    if (visited_set.get(neighborId))
                        continue;
                    selected_edges.emplace_back(neighborId);
                    if (selected_edges.size() == edge_limit)
                        return selected_edges;
                }

            } while (cur_node->lbound < ql || cur_node->rbound > qr);

            return selected_edges;
        }

        std::priority_queue<PFI> TopDown_nodeentries_search(std::vector<TreeNode *> &filterednodes, const void *query_data, int ef, int query_k, int QL, int QR, int edge_limit)
        {
            // To fix the starting points for different 'ef' parameter, set seed to a fixed number, e.g., seed =0
            // unsigned seed = 0;
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set;
            std::priority_queue<PFI> top_candidates;
            searcher::Bitset<uint64_t> visited_set(max_elements_);

            const float* query_vector = static_cast<const float*>(query_data);

            for (auto u : filterednodes)
            {
                std::uniform_int_distribution<int> u_start(u->lbound, u->rbound);
                int pid = u_start(e);
                visited_set.set(pid);
                float dis;
                if (use_pq_) {
                    dis = compute_pq_distance(query_vector, pid);
                } else {
                    char *ep_data = getDataByInternalId(pid);
                    dis = fstdistfunc_(query_data, ep_data, dist_func_param_);
                }
                candidate_set.emplace(dis, pid);
                top_candidates.emplace(dis, pid);
            }

            float lowerBound = top_candidates.top().first;

            while (!candidate_set.empty())
            {
                auto current_point_pair = candidate_set.top();

                #pragma omp atomic
                ++metric_hops;
                if (current_point_pair.first > lowerBound)
                {
                    break;
                }
                candidate_set.pop();
                int current_pid = current_point_pair.second;
                auto selected_edges = SelectEdge(current_pid, QL, QR, edge_limit, visited_set);
                int num_edges = selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    if (!use_pq_) {
                        memory::mem_prefetch_L1(getDataByInternalId(selected_edges[i]), this->prefetch_lines);
                    }
                }
                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];

                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    
                    float dis;
                    if (use_pq_) {
                        dis = compute_pq_distance(query_vector, neighbor_id);
                    } else {
                        char *neighbor_data = getDataByInternalId(neighbor_id);
                        dis = fstdistfunc_(query_data, neighbor_data, dist_func_param_);
                    }
                    
                    #pragma omp atomic
                    ++metric_distance_computations;

                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
            while (top_candidates.size() > query_k)
                top_candidates.pop();
            return top_candidates;
        }

        void search(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit)
        {
            for (auto range : storage->query_range)
            {
                int suffix = range.first;
                std::vector<std::vector<int>> &gt = storage->groundtruth[suffix];
                std::string savepath = saveprefix + std::to_string(suffix) + ".csv";
                CheckPath(savepath);
                std::ofstream outfile(savepath);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);

                // Performance metrics for different ef values
                std::vector<int> HOP;      // Average hops per query
                std::vector<int> DCO;      // Average distance computations per query
                std::vector<float> QPS;    // Queries per second
                std::vector<float> RECALL; // Recall accuracy

                outfile << "SearchEF,Recall,QPS,DCO,HOP,RAM_MB\n";
                std::cout << "suffix = " << suffix << std::endl;
                for (auto ef : SearchEF)
                {
                    int tp = 0;

                    metric_hops = 0;
                    metric_distance_computations = 0;                    
                    // Measure wall-clock time (not cumulative thread time)
                    timeval t1, t2;
                    gettimeofday(&t1, NULL);
                    
                    # pragma omp parallel for reduction(+:tp)

                    for (int i = 0; i < storage->query_nb; i++)
                    {
                        auto rp = range.second[i];
                        int ql = rp.first, qr = rp.second;

                        std::vector<TreeNode *> filterednodes = tree->range_filter(tree->root, ql, qr);
                        std::priority_queue<PFI> res = TopDown_nodeentries_search(filterednodes, storage->query_points[i].data(), ef, storage->query_K, ql, qr, edge_limit);
                        
                        std::map<int, int> record;
                        while (res.size())
                        {
                            auto x = res.top().second;
                            res.pop();
                            if (record.count(x))
                                throw Exception("repetitive search results");
                            record[x] = 1;
                            if (std::find(gt[i].begin(), gt[i].end(), x) != gt[i].end())
                                tp++;
                        }
                    }
                    
                    gettimeofday(&t2, NULL);
                    float searchtime = GetTime(t1, t2);

                    float recall = 1.0 * tp / storage->query_nb / storage->query_K;
                    float qps = storage->query_nb / searchtime;
                    float dco = metric_distance_computations * 1.0 / storage->query_nb;
                    float hop = metric_hops * 1.0 / storage->query_nb;

                    HOP.emplace_back(hop);
                    DCO.emplace_back(dco);
                    QPS.emplace_back(qps);
                    RECALL.emplace_back(recall);
                }
                struct rusage usage;
                getrusage(RUSAGE_SELF, &usage);
                long ram_mb = usage.ru_maxrss / 1024;

                for (int i = 0; i < RECALL.size(); i++)
                {
                    outfile << SearchEF[i] << "," << RECALL[i] << "," << QPS[i] << "," << DCO[i] << "," << HOP[i] << "," << ram_mb << std::endl;
                }
                outfile.close();
            }
        }

    };
}