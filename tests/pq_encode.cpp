#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>
#include "iRG_search.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

std::unordered_map<std::string, std::string> paths;

struct CompressedIndexData {
    std::unique_ptr<faiss::ProductQuantizer> pq_model;
    PQCodesBlob data_codes_blob;
    
    CompressedIndexData() = default;
    
    CompressedIndexData(const std::string& pq_model_path, const std::string& pq_codes_path) {
        LoadPQModel(pq_model_path);
        data_codes_blob = LoadPQCodes(pq_codes_path, "data");
    }
    
    void LoadPQModel(const std::string& path) {
        pq_model.reset(faiss::read_ProductQuantizer(path.c_str()));
        if (!pq_model) {
            throw std::runtime_error("failed to load PQ model from " + path);
        }
        std::cout << "Loaded PQ model: d=" << pq_model->d << ", M=" << pq_model->M 
                  << ", nbits=" << pq_model->nbits << ", code_size=" << pq_model->code_size << std::endl;
    }
    
    PQCodesBlob LoadPQCodes(const std::string& path, const std::string& label);
};

namespace {

struct DenseMatrix {
    int32_t n = 0;
    int32_t d = 0;
    std::vector<float> values;
    };

void ensure_parent_dir(const std::string& file_path) {
    std::filesystem::path p(file_path);
    auto parent = p.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

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

std::vector<float> sample_training_vectors(const DenseMatrix& data, int train_size) {
    const int actual = std::min(train_size, data.n);
    std::vector<float> sample(static_cast<size_t>(actual) * data.d);

    if (actual == data.n) {
        sample = data.values;
        return sample;
    }

    for (int i = 0; i < actual; ++i) {
        const int src_idx = static_cast<int>((static_cast<int64_t>(i) * data.n) / actual);
        const float* src = data.values.data() + static_cast<size_t>(src_idx) * data.d;
        float* dst = sample.data() + static_cast<size_t>(i) * data.d;
        std::copy(src, src + data.d, dst);
    }

    return sample;
}

void save_pq_codes(
        const std::string& path,
        int32_t n,
        int32_t d,
        int32_t M,
        int32_t nbits,
        int32_t code_size,
        const std::vector<uint8_t>& codes) {
    ensure_parent_dir(path);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open " + path);
    }

    out.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&d), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&M), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&nbits), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&code_size), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
    if (!out) {
        throw std::runtime_error("failed to write PQ codes to " + path);
    }
}

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

void TrainAndCompressPQ(const std::unordered_map<std::string, std::string>& paths, int M) {
    std::cout << "\n=== Training PQ Model and Compressing Data ===" << std::endl;

    // Load data
    std::cout << "Loading data: " << paths.at("data_vector_comp") << std::endl;
    DenseMatrix data = load_irange_bin(paths.at("data_vector_comp"));
    std::cout << "Loaded " << data.n << " vectors, dim=" << data.d << std::endl;

    // Validate dimension
    if (data.d % M != 0) {
        throw std::runtime_error("dimension must be divisible by M for ProductQuantizer");
    }

    // Parse nbits
    if (!paths.count("nbits") || paths.at("nbits").empty()) {
        throw std::runtime_error("missing --nbits argument");
    }
    int nbits = std::stoi(paths.at("nbits"));
    if (nbits <= 0 || nbits > 16) {
        throw std::runtime_error("--nbits must be in the range [1, 16]");
    }

    std::cout << "Training PQ with " << data.n << " vectors, M=" << M << ", nbits=" << nbits << std::endl;

    faiss::ProductQuantizer pq(data.d, M, nbits);
    pq.verbose = true;
    pq.train(data.n, data.values.data());

    // Save PQ model
    ensure_parent_dir(paths.at("pq_model_out"));
    faiss::write_ProductQuantizer(&pq, paths.at("pq_model_out").c_str());
    std::cout << "Saved PQ model to: " << paths.at("pq_model_out") << std::endl;

    // Compress data vectors
    std::vector<uint8_t> data_codes(static_cast<size_t>(data.n) * pq.code_size);
    pq.compute_codes(data.values.data(), data_codes.data(), data.n);
    save_pq_codes(paths.at("pq_codes_out"), data.n, data.d, M, nbits,
                  static_cast<int32_t>(pq.code_size), data_codes);
    std::cout << "Saved data PQ codes to: " << paths.at("pq_codes_out") << std::endl;

    const double raw_bytes = static_cast<double>(data.n) * data.d * sizeof(float);
    const double pq_bytes = static_cast<double>(data_codes.size());
    std::cout << "Data compression: " << (raw_bytes / pq_bytes) << "x ("
              << raw_bytes / (1024.0 * 1024.0) << " MB -> "
              << pq_bytes / (1024.0 * 1024.0) << " MB)" << std::endl;

    std::cout << "=== PQ Training and Compression Complete ===" << std::endl;
}

void InitializeIndexWithCompressedData(
        iRangeGraph::iRangeGraph_Search<float>& index,
        CompressedIndexData& compressed_data) {
    std::cout << "Setting up index with compressed data:" << std::endl;
    std::cout << "  - Codebook: " << compressed_data.pq_model->M << " subspaces, "
              << "ksub=" << compressed_data.pq_model->ksub << std::endl;
    std::cout << "  - Compressed codes: " << compressed_data.data_codes_blob.n << " vectors, "
              << "code_size=" << compressed_data.data_codes_blob.code_size << " bytes" << std::endl;
    
    index.load_pq_codes(compressed_data.data_codes_blob.codes,
                        compressed_data.data_codes_blob.code_size,
                        compressed_data.data_codes_blob.M,
                        compressed_data.data_codes_blob.nbits,
                        compressed_data.pq_model.get());
    
    const double pq_total_bytes = static_cast<double>(compressed_data.data_codes_blob.codes.size());
    const double raw_total_bytes = static_cast<double>(compressed_data.data_codes_blob.n) *
                                    compressed_data.data_codes_blob.d * sizeof(float);
    std::cout << "Compression: " << (raw_total_bytes / pq_total_bytes) << "x" << std::endl;
    std::cout << "Distance: raw query vector + compressed DB codes (per-subspace computation, no precomputed table)" << std::endl;
}


} // namespace

// Implementation of CompressedIndexData::LoadPQCodes
inline PQCodesBlob CompressedIndexData::LoadPQCodes(const std::string& path, const std::string& label) {
    PQCodesBlob blob = load_pq_codes(path);
    std::cout << "Loaded " << label << " PQ codes: n=" << blob.n << ", d=" << blob.d
              << ", M=" << blob.M << ", nbits=" << blob.nbits
              << ", code_size=" << blob.code_size << ", total_size=" 
              << (blob.codes.size() / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Query vectors: loaded raw/uncompressed from storage" << std::endl;
    return blob;
}

int main(int argc, char** argv) {
    try {
        int M_graf_edges = 32;
        int M_compression_spaces = 16;
        int query_K = 10;

        for (int i = 0; i < argc; i++)
        {
            std::string arg = argv[i];
            if (arg == "--data_path_comp")
                paths["data_vector_comp"] = argv[i + 1];
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
            if (arg == "--M_edges_graf")
                M_graf_edges = std::stoi(argv[i + 1]);
            if (arg == "--pq_codes_out")
                paths["pq_codes_out"] = argv[i + 1];
            if (arg == "--pq_model_out")
                paths["pq_model_out"] = argv[i + 1];
            if (arg == "--nbits")
                paths["nbits"] = argv[i + 1];
            if (arg == "--M_compression_spaces")
                paths["M_compression_spaces"] = argv[i + 1];
        }

        const bool model_exists = std::filesystem::exists(paths["pq_model_out"]);
        const bool data_codes_exist = std::filesystem::exists(paths["pq_codes_out"]);
        const bool query_requested = !paths["query_vector"].empty();
        const bool query_codes_exist = !query_requested || std::filesystem::exists(paths["query_codes_out"]);

        if (!(model_exists && data_codes_exist && query_codes_exist)) {
            if (paths.count("M_compression_spaces")) {
                M_compression_spaces = std::stoi(paths["M_compression_spaces"]);
                if (M_compression_spaces <= 0) {
                    throw std::runtime_error("--M_compression_spaces must be above 0");
                }
            }
            TrainAndCompressPQ(paths, M_compression_spaces);
        }

    
        std::cout << "Loading compressed index data..." << std::endl;
        
        // Load PQ model (codebook) and compressed DB codes
        CompressedIndexData compressed_data(paths["pq_model_out"], paths["pq_codes_out"]);

        iRangeGraph::DataLoader storage;
        storage.query_K = query_K;
        // Load raw query vectors (uncompressed) for search
        storage.LoadQuery(paths["query_vector"]);
        storage.LoadQueryRange(paths["range_saveprefix"]);
        storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);

        // Create index with both raw data and PQ codes
        iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector_comp"], paths["index"], &storage, M_graf_edges);
        
        // Set up the index with compressed DB vectors and codebook
        InitializeIndexWithCompressedData(index, compressed_data);
        
        // searchefs can be adjusted
        std::vector<int> SearchEF = {1700, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10};
        index.use_pq_ = true; // Enable PQ-based distance computation
        index.search(SearchEF, paths["result_saveprefix"], M_graf_edges);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "pq_encode error: " << e.what() << std::endl;
        return 1;
    }
}