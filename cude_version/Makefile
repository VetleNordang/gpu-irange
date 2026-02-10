NVCC = nvcc
ARCH = -arch=sm_60
INCLUDE = -I../include
BUILD_DIR = build
TARGET = $(BUILD_DIR)/hello
SOURCE = hello_clean.cu
HEADERS = gpu_search.cuh ../include/iRG_search.h ../include/utils.h
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

run: $(TARGET)
	$(TARGET) $(ARGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run clean
