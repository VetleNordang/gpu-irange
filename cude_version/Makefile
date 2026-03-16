NVCC = nvcc
ARCH = -arch=sm_60
INCLUDE = -I../include
PROJECT_ROOT = ..
DIST_THREADS_PER_NEIGHBOR ?= 4
OPT_FLAGS = -O3 --use_fast_math -lineinfo -DDIST_THREADS_PER_NEIGHBOR=$(DIST_THREADS_PER_NEIGHBOR) -Xcompiler "-O3 -march=native -fopenmp -DNDEBUG"
BUILD_DIR = build
TARGET = $(BUILD_DIR)/hello
OPT_TARGET = $(BUILD_DIR)/optimized_test
SOURCE = hello.cu
HEADERS = gpu_search.cuh gpu_search_updated.cuh ../include/iRG_search.h ../include/utils.h

# ARGS = --data_path $(PROJECT_ROOT)/exectable_data/video/youtube_rgb_sorted.bin \
#        --query_path $(PROJECT_ROOT)/exectable_data/video/youtube_rgb_query.bin \
#        --range_saveprefix $(PROJECT_ROOT)/exectable_data/video/query_ranges/query_ranges \
#        --groundtruth_saveprefix $(PROJECT_ROOT)/exectable_data/video/ground_truth/ground_truth \
#        --index_file $(PROJECT_ROOT)/exectable_data/video/youtube_rgb.index \
#        --result_saveprefix $(PROJECT_ROOT)/exectable_data/video/results/results \
#        --M 32

# ARGS = --data_path $(PROJECT_ROOT)/exectable_data/gist1m/250k/gist_base_250k.bin \
# 		--query_path $(PROJECT_ROOT)/exectable_data/gist1m/250k/gist_query_250k.bin \
# 		--range_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/250k/query_ranges/query_ranges_250k \
# 		--groundtruth_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/250k/groundtruth/groundtruth_250k \
# 		--index_file $(PROJECT_ROOT)/exectable_data/gist1m/250k/gist_250k.index \
# 		--result_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/250k/results/results_250k \
# 		--M 32

# ARGS = --data_path $(PROJECT_ROOT)/exectable_data/gist1m/500k/gist_base_500k.bin \
# 		--query_path $(PROJECT_ROOT)/exectable_data/gist1m/500k/gist_query_500k.bin \
# 		--range_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/500k/query_ranges/query_ranges_500k \
# 		--groundtruth_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/500k/groundtruth/groundtruth_500k \
# 		--index_file $(PROJECT_ROOT)/exectable_data/gist1m/500k/gist_500k.index \
# 		--result_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/500k/results/results_500k \
# 		--M 32

# ARGS = --data_path $(PROJECT_ROOT)/exectable_data/gist1m/750k/gist_base_750k.bin \
# 		--query_path $(PROJECT_ROOT)/exectable_data/gist1m/750k/gist_query_750k.bin \
# 		--range_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/750k/query_ranges/query_ranges_750k \
# 		--groundtruth_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/750k/groundtruth/groundtruth_750k \
# 		--index_file $(PROJECT_ROOT)/exectable_data/gist1m/750k/gist_750k.index \
# 		--result_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/750k/results/results_750k \
# 		--M 32

ARGS = --data_path $(PROJECT_ROOT)/exectable_data/gist1m/1000k/gist_base_1000k.bin \
		--query_path $(PROJECT_ROOT)/exectable_data/gist1m/1000k/gist_query_1000k.bin \
		--range_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/1000k/query_ranges/query_ranges_1000k \
		--groundtruth_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/1000k/groundtruth/groundtruth_1000k \
		--index_file $(PROJECT_ROOT)/exectable_data/gist1m/1000k/gist_1000k.index \
		--result_saveprefix $(PROJECT_ROOT)/exectable_data/gist1m/1000k/results/results_1000k \
		--M 32

# ARGS = --data_path $(PROJECT_ROOT)/exectable_data/audi/yt_aud_sorted_vec_by_attr.bin \
#        --query_path $(PROJECT_ROOT)/exectable_data/audi/yt_aud_ranged_queries.bin \
#        --range_saveprefix $(PROJECT_ROOT)/exectable_data/audi/query_ranges/query_ranges \
#        --groundtruth_saveprefix $(PROJECT_ROOT)/exectable_data/audi/ground_truth/ground_truth \
#        --index_file $(PROJECT_ROOT)/exectable_data/audi/yt_aud_irangegraph_M32.bin \
#        --result_saveprefix $(PROJECT_ROOT)/exectable_data/audi/results/results \
#        --M 32


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

run_optimized_test:
	$(MAKE) optimized_test DIST_THREADS_PER_NEIGHBOR=$(DIST_THREADS_PER_NEIGHBOR) -B
	$(OPT_TARGET) $(ARGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run optimized_test run_optimized_test clean
