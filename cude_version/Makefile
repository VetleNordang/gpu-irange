NVCC = nvcc
ARCH = -arch=sm_60
INCLUDE = -I../include
OPT_FLAGS = -O3 --use_fast_math -lineinfo -Xcompiler "-O3 -march=native -fopenmp -DNDEBUG"
BUILD_DIR = build
TARGET = $(BUILD_DIR)/hello
OPT_TARGET = $(BUILD_DIR)/optimized_test
SOURCE = hello.cu
HEADERS = gpu_search.cuh ../include/iRG_search.h ../include/utils.h

# ARGS = --data_path ../exectable_data/video/youtube_rgb_sorted.bin \
#        --query_path ../exectable_data/video/youtube_rgb_query.bin \
#        --range_saveprefix ../exectable_data/video/query_ranges/query_ranges \
#        --groundtruth_saveprefix ../exectable_data/video/ground_truth/ground_truth \
#        --index_file ../exectable_data/video/youtube_rgb.index \
#        --result_saveprefix ../exectable_data/video/results/results \
#        --M 32
ARGS = --data_path ../exectable_data/audi/yt_aud_sorted_vec_by_attr.bin \
       --query_path ../exectable_data/audi/yt_aud_ranged_queries.bin \
       --range_saveprefix ../exectable_data/audi/query_ranges/query_ranges \
       --groundtruth_saveprefix ../exectable_data/audi/ground_truth/ground_truth \
       --index_file ../exectable_data/audi/yt_aud_irangegraph_M32.bin \
       --result_saveprefix ../exectable_data/audi/results/results \
       --M 32

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(BUILD_DIR) $(SOURCE) $(HEADERS)
	$(NVCC) $(ARCH) $(INCLUDE) $(SOURCE) -o $(TARGET)

optimized_test: $(OPT_TARGET)

$(OPT_TARGET): $(BUILD_DIR) $(SOURCE) $(HEADERS)
	$(NVCC) $(ARCH) $(OPT_FLAGS) $(INCLUDE) $(SOURCE) -o $(OPT_TARGET)

run: $(TARGET)
	$(TARGET) $(ARGS)

run_optimized_test: $(OPT_TARGET)
	$(OPT_TARGET) $(ARGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run optimized_test run_optimized_test clean
