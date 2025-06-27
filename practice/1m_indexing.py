import numpy as np
import time
import os
from diskannpy import build_disk_index, StaticDiskIndex

# --- 설정 ---
num_vectors = 1_000_000
dim = 1024
K = 10
index_dir = "diskann_index"
os.makedirs(index_dir, exist_ok=True)

# --- 벡터 생성 (float32) ---
print("Generating random vectors...")
data = np.random.rand(num_vectors, dim).astype(np.float32)
query = data[:10]  # 검색 테스트용 쿼리 10개

# --- 벡터 저장 ---
data_bin_path = os.path.join(index_dir, "base.bin")
data.tofile(data_bin_path)

# --- 인덱스 생성 ---
print("Building index...")
start = time.time()
build_disk_index(
    data=data,
    distance_metric="l2",
    index_directory=index_dir,
    complexity=64,
    graph_build_threads=os.cpu_count(),
    search_memory_budget=2 * 1024,  # MB
    build_memory_budget=8 * 1024,   # MB
    alpha=1.2,
    num_threads=os.cpu_count(),
    pq_disk_bytes=0  # disable PQ compression for simplicity
)
build_time = time.time() - start
print(f"Index built in {build_time:.2f} seconds")

# --- 인덱스 로드 및 검색 ---
print("Loading index for search...")
index = StaticDiskIndex(index_dir)

print("Running queries...")
start = time.time()
neighbors, distances = index.search(query, K)
search_time = time.time() - start

print(f"Search time for {len(query)} queries: {search_time:.2f} seconds")
print(f"Avg latency per query: {search_time / len(query) * 1000:.2f} ms")
print("Top-1 neighbor indices for each query:", neighbors[:, 0])
