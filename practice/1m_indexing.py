import numpy as np
import time
import os
from diskannpy import build_disk_index, StaticDiskIndex

# --- 설정 ---
num_vectors = 100_000
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
    graph_degree=64,                # ✅ 새로 추가
    search_memory_maximum=2.0,      # ✅ GB 단위로 변경
    build_memory_maximum=8.0,       # ✅ GB 단위로 변경
    num_threads=os.cpu_count(),
    pq_disk_bytes=0
)
build_time = time.time() - start
print(f"Index built in {build_time:.2f} seconds")

# --- 인덱스 로드 및 검색 ---
print("Loading index for search...")
index = StaticDiskIndex(index_directory=index_dir,num_threads=os.cpu_count(),
    num_nodes_to_cache=10000)

all_neighbors = []
all_distances = []


print("Running queries...")
start = time.time()
for q in query:  # query: shape = (10, 1024)
    neighbors, distances = index.search(q, k_neighbors=10, complexity=100)
    all_neighbors.append(neighbors)
    all_distances.append(distances)

search_time = time.time() - start

import numpy as np
all_neighbors = np.stack(all_neighbors)
all_distances = np.stack(all_distances)

print(f"Search time for {len(query)} queries: {search_time:.2f} seconds")
print(f"Avg latency per query: {search_time / len(query) * 1000:.2f} ms")
print("Top-1 neighbor indices for each query:", all_neighbors[:, 0])
