#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

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
        throw std::runtime_error("cannot open: " + path);
    }

    DenseMatrix mat;
    in.read(reinterpret_cast<char*>(&mat.n), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&mat.d), sizeof(int32_t));
    if (!in || mat.n <= 0 || mat.d <= 0) {
        throw std::runtime_error("invalid matrix header in: " + path);
    }

    mat.values.resize(static_cast<size_t>(mat.n) * mat.d);
    in.read(reinterpret_cast<char*>(mat.values.data()), mat.values.size() * sizeof(float));
    if (!in) {
        throw std::runtime_error("failed to read matrix data from: " + path);
    }
    return mat;
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
        throw std::runtime_error("cannot open for writing: " + path);
    }

    out.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&d), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&M), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&nbits), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&code_size), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
    if (!out) {
        throw std::runtime_error("failed to write PQ codes to: " + path);
    }
}

int main(int argc, char** argv) {
    try {
        // Parse command-line arguments
        std::string data_path;
        std::string model_out_path;
        std::string codes_out_path;
        int M = 16;
        int nbits = 8;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--data" && i + 1 < argc) {
                data_path = argv[++i];
            } else if (arg == "--model_out" && i + 1 < argc) {
                model_out_path = argv[++i];
            } else if (arg == "--codes_out" && i + 1 < argc) {
                codes_out_path = argv[++i];
            } else if (arg == "--M" && i + 1 < argc) {
                M = std::stoi(argv[++i]);
            } else if (arg == "--nbits" && i + 1 < argc) {
                nbits = std::stoi(argv[++i]);
            }
        }

        // Validate required arguments
        if (data_path.empty()) {
            std::cerr << "Error: --data <path> is required" << std::endl;
            return 1;
        }
        if (model_out_path.empty()) {
            std::cerr << "Error: --model_out <path> is required" << std::endl;
            return 1;
        }
        if (codes_out_path.empty()) {
            std::cerr << "Error: --codes_out <path> is required" << std::endl;
            return 1;
        }
        if (M <= 0) {
            std::cerr << "Error: --M must be positive (got " << M << ")" << std::endl;
            return 1;
        }
        if (nbits <= 0 || nbits > 16) {
            std::cerr << "Error: --nbits must be in [1, 16] (got " << nbits << ")" << std::endl;
            return 1;
        }

        std::cout << "=== PQ Training and Compression ===" << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  Data input: " << data_path << std::endl;
        std::cout << "  Model output: " << model_out_path << std::endl;
        std::cout << "  Codes output: " << codes_out_path << std::endl;
        std::cout << "  M (subspaces): " << M << std::endl;
        std::cout << "  nbits: " << nbits << std::endl;
        std::cout << std::endl;

        // Load data
        std::cout << "Loading data vectors..." << std::endl;
        DenseMatrix data = load_irange_bin(data_path);
        std::cout << "  Loaded: " << data.n << " vectors, dimension=" << data.d << std::endl;

        // Validate dimension
        if (data.d % M != 0) {
            throw std::runtime_error("Dimension (" + std::to_string(data.d) + 
                                     ") must be divisible by M (" + std::to_string(M) + ")");
        }

        // Create and train PQ model
        std::cout << "\nTraining PQ model..." << std::endl;
        faiss::ProductQuantizer pq(data.d, M, nbits);
        pq.verbose = true;
        pq.train(data.n, data.values.data());

        // Save PQ model
        std::cout << "\nSaving PQ model..." << std::endl;
        ensure_parent_dir(model_out_path);
        faiss::write_ProductQuantizer(&pq, model_out_path.c_str());
        std::cout << "  Saved to: " << model_out_path << std::endl;

        // Compress all data
        std::cout << "\nCompressing data vectors..." << std::endl;
        std::vector<uint8_t> data_codes(static_cast<size_t>(data.n) * pq.code_size);
        pq.compute_codes(data.values.data(), data_codes.data(), data.n);

        // Save compressed codes
        std::cout << "Saving compressed codes..." << std::endl;
        save_pq_codes(codes_out_path, data.n, data.d, M, nbits,
                      static_cast<int32_t>(pq.code_size), data_codes);
        std::cout << "  Saved to: " << codes_out_path << std::endl;

        // Print compression statistics
        const double raw_bytes = static_cast<double>(data.n) * data.d * sizeof(float);
        const double pq_bytes = static_cast<double>(data_codes.size());
        const double ratio = raw_bytes / pq_bytes;

        std::cout << "\nCompression Statistics:" << std::endl;
        std::cout << "  Raw size: " << std::fixed << std::setprecision(2) 
                  << raw_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Compressed size: " << pq_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Compression ratio: " << ratio << "x" << std::endl;

        std::cout << "\n=== SUCCESS ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
