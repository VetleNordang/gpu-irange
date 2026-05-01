source /cluster/home/vetlean/.venv_yt8m/bin/activate
module load GCCcore/11.3.0
module load CUDA/12.1.1
module load Anaconda3/2024.02-1

CONDA_FAISS=/cluster/home/vetlean/.conda/envs/irange
export FAISS_INCLUDE="-I$CONDA_FAISS/include"
export FAISS_LIB_PATH="-L$CONDA_FAISS/lib"
export LD_LIBRARY_PATH="$CONDA_FAISS/lib:$LD_LIBRARY_PATH"
export PATH="$CONDA_FAISS/bin:$PATH"
export CONDA_PYTHON="$CONDA_FAISS/bin/python3"

# Modify hello.cu with debug prints
sed -i 's/CheckPath(savepath);/std::cout << "Calling CheckPath..." << std::endl; CheckPath(savepath); std::cout << "CheckPath done." << std::endl;/g' cude_version/hello.cu
sed -i 's/int query_nb = index.storage->query_nb;/std::cout << "Init visited array..." << std::endl; int query_nb = index.storage->query_nb;/g' cude_version/hello.cu

make -C cude_version optimized_test FAISS_INCLUDE="$FAISS_INCLUDE" FAISS_LIB_PATH="$FAISS_LIB_PATH" -j4
./cude_version/build/optimized_test 1 --index exectable_data/video/1m/index --groundtruth exectable_data/video/1m/ground_truth/ --queries exectable_data/video/1m/queries.bin --saveprefix exectable_data/video/1m/ranges10 --ef 10,20,30 --K 10 --threads 32
