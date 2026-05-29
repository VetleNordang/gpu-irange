#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

// TFRecord format: [uint64 length][uint32 masked_crc32_of_length][data][uint32 masked_crc32_of_data]
static long count_records(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return -1;

    long count = 0;
    uint64_t length;
    uint32_t crc;

    while (fread(&length, 8, 1, f) == 1) {
        fread(&crc, 4, 1, f);       // crc of length
        fseek(f, (long)length, SEEK_CUR); // skip data
        fread(&crc, 4, 1, f);       // crc of data
        count++;
    }

    fclose(f);
    return count;
}

static std::vector<std::string> glob_prefix(const fs::path& dir, const std::string& prefix) {
    std::vector<std::string> result;
    for (auto& entry : fs::directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && name.size() > 9 &&
            name.substr(name.size() - 9) == ".tfrecord") {
            result.push_back(entry.path().string());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

int main() {
    const fs::path dir = "/workspaces/irange/executable_data/yt_tf";

    struct Partition { std::string name; std::string prefix; };
    std::vector<Partition> partitions = {
        {"train",    "train"},
        {"validate", "validate"},
        {"test",     "test"},
    };

    long grand_total = 0;

    for (auto& p : partitions) {
        auto files = glob_prefix(dir, p.prefix);
        long count = 0;

        for (size_t i = 0; i < files.size(); i++) {
            long n = count_records(files[i]);
            if (n < 0) {
                fprintf(stderr, "failed to open %s\n", files[i].c_str());
                continue;
            }
            count += n;

            if ((i + 1) % 100 == 0) {
                printf("%s: %zu/%zu shards, %ld embeddings so far\n",
                       p.name.c_str(), i + 1, files.size(), count);
                fflush(stdout);
            }
        }

        printf("%s: %ld embeddings across %zu shards\n",
               p.name.c_str(), count, files.size());
        grand_total += count;
    }

    printf("\nTotal: %ld embeddings\n", grand_total);
    return 0;
}
