NVCC = nvcc
GPU_SM ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
GPU_SM := $(or $(GPU_SM),80)
ARCH = -gencode arch=compute_$(GPU_SM),code=sm_$(GPU_SM)
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
ROOT_SOURCE = hello_root.cu
ROOT_TARGET = $(BUILD_DIR)/hello_root
HEADERS = gpu_search.cuh gpu_search_updated.cuh gpu_search_pq.cuh gpu_pq_distance.cuh gpu_index.cuh gpu_heap.cuh gpu_visited.cuh ../include/iRG_search.h ../include/utils.h

# FAISS support
ENABLE_FAISS ?= 1
ifeq ($(ENABLE_FAISS),1)
	FAISS_CFLAGS = -DENABLE_FAISS
	FAISS_LDFLAGS = -lfaiss -lgomp -lblas -llapack
endif

# Normal distance computation args (audi 1m dataset)
ARGS = --data_path $(PROJECT_ROOT)/executable_data/audi/1m/yt_aud_1m.bin \
		--query_path $(PROJECT_ROOT)/executable_data/audi/1m/yt_aud_query.bin \
		--range_saveprefix $(PROJECT_ROOT)/executable_data/audi/1m/query_ranges/qr \
		--groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/audi/1m/groundtruth/gt \
		--index_file $(PROJECT_ROOT)/executable_data/audi/1m/yt_aud_1m.index \
		--result_saveprefix $(PROJECT_ROOT)/executable_data/audi/1m/results/results_1m_ \
		--M 32

# ARGS = --data_path $(PROJECT_ROOT)/executable_data/gist1m/1000k/gist_base_1000k.bin \
# 		--query_path $(PROJECT_ROOT)/executable_data/gist1m/1000k/gist_query_1000k.bin \
# 		--range_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/1000k/query_ranges/query_ranges_1000k \
# 		--groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/1000k/groundtruth/groundtruth_1000k \
# 		--index_file $(PROJECT_ROOT)/executable_data/gist1m/1000k/gist_1000k.index \
# 		--result_saveprefix $(PROJECT_ROOT)/executable_data/gist1m/1000k/results/results_1000k_ \
# 		--M 32


PQ_ARGS = --data_path_comp $(PROJECT_ROOT)/executable_data/audi/2m/yt_aud_2m.bin \
		--query_path $(PROJECT_ROOT)/executable_data/audi/2m/yt_aud_query.bin \
		--index_path $(PROJECT_ROOT)/executable_data/audi/2m/yt_aud_2m.index \
		--result_saveprefix $(PROJECT_ROOT)/executable_data/audi/2m/results/results_pq \
		--range_saveprefix $(PROJECT_ROOT)/executable_data/audi/2m/query_ranges/qr \
		--groundtruth_saveprefix $(PROJECT_ROOT)/executable_data/audi/2m/groundtruth/gt \
		--pq_model_out $(PROJECT_ROOT)/executable_data/audi/2m/pq/audi_2m_pq_m32_nb9.faiss \
		--pq_codes_out $(PROJECT_ROOT)/executable_data/audi/2m/pq/audi_2m_pq_codes_m32_nb9.bin \
		--M_compression_spaces 32 \
		--graph_M 32




all: $(TARGET)

root_target: $(ROOT_TARGET)

pq_target: $(PQ_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(BUILD_DIR) $(SOURCE) $(HEADERS)
	$(NVCC) $(ARCH) $(INCLUDE) $(FAISS_CFLAGS) -Xcompiler "-DENABLE_FAISS,-fopenmp" -Xcompiler "-lgomp" $(SOURCE) -o $(TARGET) $(FAISS_LIB_PATH) -lgomp -lblas -llapack -lfaiss

$(PQ_TARGET): $(BUILD_DIR) $(PQ_SOURCE) $(HEADERS)
	$(NVCC) $(ARCH) $(OPT_FLAGS) $(INCLUDE) $(FAISS_CFLAGS) -Xcompiler "$(FAISS_CFLAGS),-fopenmp,-lgomp" $(PQ_SOURCE) -o $(PQ_TARGET) $(FAISS_LIB_PATH) -lgomp -lblas -llapack -lfaiss

$(ROOT_TARGET): $(BUILD_DIR) $(ROOT_SOURCE) $(HEADERS)
	$(NVCC) $(ARCH) $(OPT_FLAGS) $(INCLUDE) -Xcompiler "-fopenmp,-lgomp" $(ROOT_SOURCE) -o $(ROOT_TARGET) -lgomp

optimized_test: $(OPT_TARGET)

$(OPT_TARGET): $(BUILD_DIR) $(SOURCE) $(HEADERS)
	$(NVCC) $(ARCH) $(OPT_FLAGS) $(INCLUDE) $(FAISS_CFLAGS) -Xcompiler "$(FAISS_CFLAGS),-fopenmp,-lgomp" $(SOURCE) -o $(OPT_TARGET) $(FAISS_LIB_PATH) -lgomp -lblas -llapack -lfaiss

run: $(TARGET)
	$(TARGET) $(ARGS)

run_pq: $(PQ_TARGET)
	$(PQ_TARGET) $(PQ_ARGS)

run_root: $(ROOT_TARGET)
	$(ROOT_TARGET) $(ARGS)

run_optimized_test:
	$(MAKE) optimized_test DIST_THREADS_PER_NEIGHBOR=$(DIST_THREADS_PER_NEIGHBOR) -B
	$(OPT_TARGET) $(ARGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all pq_target root_target run run_pq run_root optimized_test run_optimized_test clean
