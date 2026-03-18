#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct DenseMatrix {
    int32_t n = 0;
    int32_t d = 0;
    std::vector<float> values;
};

DenseMatrix load_irange_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open " + path);
    }

    DenseMatrix mat;
    in.read(reinterpret_cast<char*>(&mat.n), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&mat.d), sizeof(int32_t));
    if (!in || mat.n <= 0 || mat.d <= 0) {
        throw std::runtime_error("invalid matrix header in " + path);
    }

    mat.values.resize(static_cast<size_t>(mat.n) * mat.d);
    in.read(reinterpret_cast<char*>(mat.values.data()), mat.values.size() * sizeof(float));
    if (!in) {
        throw std::runtime_error("failed to read matrix payload from " + path);
    }
    return mat;
}

struct PQCodesBlob {
    int32_t n = 0;
    int32_t d = 0;
    int32_t M = 0;
    int32_t nbits = 0;
    int32_t code_size = 0;
    std::vector<uint8_t> codes;
};

PQCodesBlob load_pq_codes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open " + path);
    }

    PQCodesBlob blob;
    in.read(reinterpret_cast<char*>(&blob.n), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&blob.d), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&blob.M), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&blob.nbits), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&blob.code_size), sizeof(int32_t));
    if (!in || blob.n <= 0 || blob.d <= 0 || blob.M <= 0 || blob.nbits <= 0 || blob.code_size <= 0) {
        throw std::runtime_error("invalid PQ code header in " + path);
    }

    const size_t payload_size = static_cast<size_t>(blob.n) * static_cast<size_t>(blob.code_size);
    blob.codes.resize(payload_size);
    in.read(reinterpret_cast<char*>(blob.codes.data()), static_cast<std::streamsize>(payload_size));
    if (!in) {
        throw std::runtime_error("failed to read PQ code payload from " + path);
    }

    return blob;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pq_model> <pq_codes> <query_bin>" << std::endl;
        return 1;
    }

    try {
        std::string pq_model_path = argv[1];
        std::string pq_codes_path = argv[2];
        std::string query_path = argv[3];

        // Load PQ model
        std::cout << "Loading PQ model from: " << pq_model_path << std::endl;
        std::unique_ptr<faiss::ProductQuantizer> pq(faiss::read_ProductQuantizer(pq_model_path.c_str()));
        if (!pq) {
            throw std::runtime_error("failed to load PQ model");
        }
        std::cout << "PQ model: d=" << pq->d << ", M=" << pq->M << ", nbits=" << pq->nbits
                  << ", ksub=" << pq->ksub << ", dsub=" << pq->dsub
                  << ", code_size=" << pq->code_size << std::endl;

        // Load compressed codes
        std::cout << "Loading PQ codes from: " << pq_codes_path << std::endl;
        PQCodesBlob codes_blob = load_pq_codes(pq_codes_path);
        std::cout << "Codes: n=" << codes_blob.n << ", d=" << codes_blob.d
                  << ", M=" << codes_blob.M << ", nbits=" << codes_blob.nbits << std::endl;

        // Load query vectors
        std::cout << "Loading queries from: " << query_path << std::endl;
        DenseMatrix queries = load_irange_bin(query_path);
        std::cout << "Queries: n=" << queries.n << ", d=" << queries.d << std::endl;

        if (queries.d != static_cast<int32_t>(pq->d)) {
            throw std::runtime_error("query dimension mismatch");
        }

        // ========== EXAMPLE: Query 0 vs Database Vector 0 ==========
        std::cout << "\n=== MANUAL DISTANCE CALCULATION EXAMPLE ===" << std::endl;
        std::cout << "Computing distance between Query 0 and Database Vector 0\n" << std::endl;

        const int qidx = 0;
        const int dbidx = 0;

        const float* query_vec = queries.values.data() + static_cast<size_t>(qidx) * queries.d;
        const uint8_t* db_code = codes_blob.codes.data() + static_cast<size_t>(dbidx) * codes_blob.code_size;

        std::cout << "Query 0 (first 10 dims): ";
        for (int i = 0; i < std::min(10, queries.d); ++i) {
            std::cout << query_vec[i] << " ";
        }
        std::cout << "...\n" << std::endl;

        std::cout << "Database Vector 0 code (all M bytes, i.e., centroid indices):" << std::endl;
        std::cout << "  ";
        for (int m = 0; m < static_cast<int>(pq->M); ++m) {
            std::cout << static_cast<int>(db_code[m]) << " ";
        }
        std::cout << "\n" << std::endl;

        // Step 1: For each subspace m, compute distance from query_m to all ksub centroids
        std::cout << "Step 1: For each subspace m, compute query_m-to-centroid distances:" << std::endl;
        std::vector<std::vector<float>> distance_table(pq->M, std::vector<float>(pq->ksub));

        for (size_t m = 0; m < pq->M; ++m) {
            const float* query_m = query_vec + m * pq->dsub;
            const float* centroids_m = pq->get_centroids(m, 0);

            for (size_t k = 0; k < pq->ksub; ++k) {
                float dist_sq = 0.0f;
                for (size_t d = 0; d < pq->dsub; ++d) {
                    float delta = query_m[d] - centroids_m[k * pq->dsub + d];
                    dist_sq += delta * delta;
                }
                distance_table[m][k] = dist_sq;
            }

            std::cout << "  Subspace " << m << " (dsub=" << pq->dsub << "): ";
            std::cout << "distances to first 5 centroids: ";
            for (size_t k = 0; k < std::min(size_t(5), pq->ksub); ++k) {
                std::cout << distance_table[m][k] << " ";
            }
            std::cout << "..." << std::endl;
        }

        // Step 2: Look up centroid indices in db_code and sum distances
        std::cout << "\nStep 2: For DB vector 0, sum distances using centroid indices:" << std::endl;
        float approx_distance_sq = 0.0f;
        for (size_t m = 0; m < pq->M; ++m) {
            uint8_t centroid_idx = db_code[m];
            float dist_m = distance_table[m][centroid_idx];
            approx_distance_sq += dist_m;

            std::cout << "  Subspace " << m << ": centroid index = " << static_cast<int>(centroid_idx)
                      << ", distance^2 = " << dist_m << std::endl;
        }

        float approx_distance = std::sqrt(approx_distance_sq);
        std::cout << "\n  Total squared distance: " << approx_distance_sq << std::endl;
        std::cout << "  Approximate L2 distance: " << approx_distance << std::endl;

        std::cout << "\n=== All components finished successfully ===" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
