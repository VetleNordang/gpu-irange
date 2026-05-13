NVCC = nvcc
ARCH = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80
FAISS_INCLUDE ?= -I/cluster/home/vetlean/.conda/envs/irange/include
FAISS_LIB_PATH ?= -L/cluster/home/vetlean/.conda/envs/irange/lib
INCLUDE = -I../include $(FAISS_INCLUDE)
PROJECT_ROOT = ..
DIST_THREADS_PER_NEIGHBOR ?= 4
OPT_FLAGS = -O3 --use_fast_math -lineinfo -DDIST_THREADS_PER_NEIGHBOR=$(DIST_THREADS_PER_NEIGHBOR) -Xcompiler "-O3 -fopenmp -DNDEBUG"
BUILD_DIR = build
TARGET = $(BUILD_DIR)/hello
OPT_TARGET = $(BUILD_DIR)/optimized_test
PQ_TARGET = $(BUILD_DIR)/hello_pq
SOURCE = hello.cu
PQ_SOURCE = hello_pq.cu
HEADERS = gpu_search.cuh gpu_search_updated.cuh ../include/iRG_search.h ../include/utils.h

# FAISS support
ENABLE_FAISS ?= 1
ifeq ($(ENABLE_FAISS),1)
	FAISS_CFLAGS = -DENABLE_FAISS
	FAISS_LDFLAGS = -lfaiss -lgomp -lblas -llapack
endif

# ARGS = --data_path $(PROJECT_ROOT)/executable_data/video/1m/youtube_rgb_1m.bin \
#        --query_path $(PROJECT_ROOT)/executable_data/video/1m/youtube_rgb_query.bin \
#        --range_saveprefix $(PROJECT_ROOT)/executable_data/video/1m/query_ranges/qr \
#        --groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/video/1m/groundtruth/gt \
#        --index_file $(PROJECT_ROOT)/executable_data/video/1m/youtube_rgb_1m.index \
#        --result_saveprefix $(PROJECT_ROOT)/executable_data/video/1m/results/results \
#        --M 32

# ARGS = --data_path $(PROJECT_ROOT)/executable_data/gist1m/250k/gist_base_250k.bin \
# 		--query_path $(PROJECT_ROOT)/executable_data/gist1m/250k/gist_query_250k.bin \
# 		--range_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/250k/query_ranges/query_ranges_250k \
# 		--groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/250k/groundtruth/groundtruth_250k \
# 		--index_file $(PROJECT_ROOT)/executable_data/gist1m/250k/gist_250k.index \
# 		--result_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/250k/results/results_250k \
# 		--M 32

# ARGS = --data_path $(PROJECT_ROOT)/executable_data/gist1m/500k/gist_base_500k.bin \
# 		--query_path $(PROJECT_ROOT)/executable_data/gist1m/500k/gist_query_500k.bin \
# 		--range_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/500k/query_ranges/query_ranges_500k \
# 		--groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/500k/groundtruth/groundtruth_500k \
# 		--index_file $(PROJECT_ROOT)/executable_data/gist1m/500k/gist_500k.index \
# 		--result_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/500k/results/results_500k \
# 		--M 32

# ARGS = --data_path $(PROJECT_ROOT)/executable_data/gist1m/750k/gist_base_750k.bin \
# 		--query_path $(PROJECT_ROOT)/executable_data/gist1m/750k/gist_query_750k.bin \
# 		--range_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/750k/query_ranges/query_ranges_750k \
# 		--groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/750k/groundtruth/groundtruth_750k \
# 		--index_file $(PROJECT_ROOT)/executable_data/gist1m/750k/gist_750k.index \
# 		--result_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/750k/results/results_750k \
# 		--M 32

# Normal distance computation args (500k dataset)
ARGS = --data_path $(PROJECT_ROOT)/executable_data/gist1m/500k/gist_base_500k.bin \
		--query_path $(PROJECT_ROOT)/executable_data/gist1m/500k/gist_query_500k.bin \
		--range_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/500k/query_ranges/query_ranges_500k_ \
		--groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/500k/groundtruth/groundtruth_500k_ \
		--index_file $(PROJECT_ROOT)/executable_data/gist1m/500k/gist_500k.index \
		--result_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/500k/results/results_500k_ \
		--M 32

# PQ compression mode args (250k dataset - working configuration)
# PQ_ARGS = --data_path_comp $(PROJECT_ROOT)/executable_data/gist1m/250k/gist_base_250k.bin \
# 		--query_path $(PROJECT_ROOT)/executable_data/gist1m/250k/gist_query_250k.bin \
# 		--index_path $(PROJECT_ROOT)/executable_data/gist1m/250k/gist_250k.index \
# 		--result_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/250k/results/results_pq \
# 		--range_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/250k/query_ranges/query_ranges_250k \
# 		--groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/250k/groundtruth/groundtruth_250k \
# 		--pq_model_out $(PROJECT_ROOT)/executable_data/gist1m/250k/pq/gist_250k_pq_m320.faiss \
# 		--pq_codes_out $(PROJECT_ROOT)/executable_data/gist1m/250k/pq/gist_250k_codes_m320.bin \
# 		--M_compression_spaces 320 \
# 		--graph_M 32

PQ_ARGS = --data_path_comp $(PROJECT_ROOT)/executable_data/gist1m/500k/gist_base_500k.bin \
		--query_path $(PROJECT_ROOT)/executable_data/gist1m/500k/gist_query_500k.bin \
		--index_path $(PROJECT_ROOT)/executable_data/gist1m/500k/gist_500k.index \
		--result_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/500k/results/pq/results_pq_ \
		--range_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/500k/query_ranges/query_ranges_500k_ \
		--groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/500k/groundtruth/groundtruth_500k_ \
		--pq_model_out $(PROJECT_ROOT)/executable_data/gist1m/500k/pq/gist_500k_pq_m320_nb9.faiss \
		--pq_codes_out $(PROJECT_ROOT)/executable_data/gist1m/500k/pq/gist_500k_pq_codes_m320_nb9.bin \
		--M_compression_spaces 320 \
		--graph_M 32

# ARGS = --data_path $(PROJECT_ROOT)/executable_data/audi/1m/yt_aud_1m.bin \
#        --query_path $(PROJECT_ROOT)/executable_data/audi/1m/yt_aud_query.bin \
#        --range_saveprefix $(PROJECT_ROOT)/executable_data/audi/1m/query_ranges/qr \
#        --groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/audi/1m/groundtruth/gt \
#        --index_file $(PROJECT_ROOT)/executable_data/audi/1m/yt_aud_1m.index \
#        --result_saveprefix $(PROJECT_ROOT)/executable_data/audi/1m/results \
#        --M 32


all: $(TARGET)

pq_target: $(PQ_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(BUILD_DIR) $(SOURCE) $(HEADERS)
	$(NVCC) $(ARCH) $(INCLUDE) $(FAISS_CFLAGS) -Xcompiler "-DENABLE_FAISS,-fopenmp" -Xcompiler "-lgomp" $(SOURCE) -o $(TARGET) $(FAISS_LIB_PATH) -lgomp -lblas -llapack -lfaiss

$(PQ_TARGET): $(BUILD_DIR) $(PQ_SOURCE) $(HEADERS)
	$(NVCC) $(ARCH) $(INCLUDE) $(FAISS_CFLAGS) -Xcompiler "-DENABLE_FAISS,-fopenmp" -Xcompiler "-lgomp" $(PQ_SOURCE) -o $(PQ_TARGET) $(FAISS_LIB_PATH) -lgomp -lblas -llapack -lfaiss

optimized_test: $(OPT_TARGET)

$(OPT_TARGET): $(BUILD_DIR) $(SOURCE) $(HEADERS)
	$(NVCC) $(ARCH) $(OPT_FLAGS) $(INCLUDE) $(FAISS_CFLAGS) -Xcompiler "$(FAISS_CFLAGS),-fopenmp,-lgomp" $(SOURCE) -o $(OPT_TARGET) $(FAISS_LIB_PATH) -lgomp -lblas -llapack -lfaiss

run: $(TARGET)
	$(TARGET) $(ARGS)

run_pq: $(PQ_TARGET)
	$(PQ_TARGET) $(PQ_ARGS)

run_optimized_test:
	$(MAKE) optimized_test DIST_THREADS_PER_NEIGHBOR=$(DIST_THREADS_PER_NEIGHBOR) -B
	$(OPT_TARGET) $(ARGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all pq_target run run_pq optimized_test run_optimized_test clean
