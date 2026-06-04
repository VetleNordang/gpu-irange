# iRangeGraph

This is a fork of [iRangeGraph](https://github.com/YuexuanXu7/iRangeGraph), the implementation of [iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search](https://arxiv.org/abs/2409.02571).

This fork extends the original with parallel CPU search and a GPU implementation using CUDA, developed as part of a Master's thesis at NTNU.

## Build

### CPU

```bash
mkdir build && cd build && cmake .. && make -j$(nproc)
```

With FAISS Product Quantization support:

```bash
cmake -DUSE_FAISS=ON -DFAISS_ROOT=/path/to/faiss .. && make -j$(nproc)
```

### GPU

Requires CUDA and FAISS. Auto-detects GPU compute capability.

```bash
cd cuda_version && make optimized_test
```

To build the PQ-compressed GPU version:

```bash
cd cuda_version && make pq_target
```

## Construct Index

#### Parameters

**`--data_path`**: Input data in .bin format. First 4 bytes: number of points. Next 4 bytes: dimension. Remaining bytes: `n*d*sizeof(float)` floats, one point at a time. Data must be pre-sorted in ascending order by attribute.

**`--index_file`**: Output path for the constructed index (.bin format).

**`--M`**: Graph degree.

**`--ef_construction`**: Size of the result set during index building.

**`--threads`**: Number of threads for index building.

#### Command

```bash
./tests/buildindex --data_path [path to data] --index_file [path to save index] --M [integer] --ef_construction [integer] --threads [integer]
```

## Search

#### Parameters

**`--data_path`**: Data points in .bin format (same as used for index construction).

**`--query_path`**: Query vectors in .bin format.

**`--index_file`**: Path to the constructed index.

**`--range_saveprefix`**: Folder where query range files will be saved. 0–9 denote range fractions 2^0, 2^-1, ..., 2^-9; 17 denotes mixed range fraction.

**`--groundtruth_saveprefix`**: Folder where groundtruth files will be saved.

**`--result_saveprefix`**: Folder where result files will be saved.

**`--M`**: Graph degree. Must match the value used during index construction.

#### CPU search

```bash
./tests/search --data_path [path to data] --query_path [path to queries] --range_saveprefix [folder] --groundtruth_saveprefix [folder] --index_file [path to index] --result_saveprefix [folder] --M [integer]
```

#### GPU search

```bash
cd cuda_version && make run
```

Or run the binary directly:

```bash
./cuda_version/build/optimized_test --data_path [path to data] --query_path [path to queries] --range_saveprefix [folder] --groundtruth_saveprefix [folder] --index_file [path to index] --result_saveprefix [folder] --M [integer]
```

#### GPU search with PQ compression

Requires FAISS. Builds a PQ model on the fly if not already present.

```bash
cd cuda_version && make run_pq
```

Or run the binary directly with additional parameters:

**`--data_path_comp`**: Compressed data path.

**`--index_path`**: Path to the HNSW index.

**`--pq_model_out`**: Output path for the trained PQ model (.faiss).

**`--pq_codes_out`**: Output path for PQ-encoded vectors (.bin).

**`--M_compression_spaces`**: Number of PQ subspaces.

**`--graph_M`**: Graph degree (must match index construction).

```bash
./cuda_version/build/gpu_pq --data_path_comp [path] --query_path [path] --index_path [path] --result_saveprefix [folder] --range_saveprefix [folder] --groundtruth_saveprefix [folder] --pq_model_out [path] --pq_codes_out [path] --M_compression_spaces [integer] --graph_M [integer]
```

## Datasets

| Dataset | Vector Type | Dimension | Attribute Type |
|---------|-------------|-----------|----------------|
| [GIST1M](http://corpus-texmex.irisa.fr/) | image descriptor | 960 | — |
| [YouTube-Audio](https://research.google.com/youtube8m/download.html) | audio | 128 | publish time, number of views |
| YouTube-Audio (Audi subset) | audio | 128 | number of likes |
