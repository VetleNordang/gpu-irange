NVCC = nvcc
ARCH = -arch=sm_60
INCLUDE = -I../include
TARGET = hello
SOURCE = hello.cu
ARGS = --data_path ../exectable_data/audi/yt_aud_sorted_vec_by_attr.bin \
       --query_path ../exectable_data/audi/yt_aud_ranged_queries.bin \
       --range_saveprefix ../exectable_data/audi/exp/query_ranges \
       --groundtruth_saveprefix ../exectable_data/audi/exp/ground_truth \
       --index_file ../exectable_data/audi/yt_aud_irangegraph_M32.bin \
       --result_saveprefix ../exectable_data/audi/exp/results \
       --M 32

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(ARCH) $(INCLUDE) $(SOURCE) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
