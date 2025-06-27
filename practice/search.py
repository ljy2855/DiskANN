import numpy as np
import time
from diskannpy import StaticDiskIndex

# --- 설정 ---
index_dir = "diskann_index"
dim = 1024
k = 10
complexity = 100
num_threads = 2
num_nodes_to_cache = 20000

# --- 쿼리 벡터 로딩 또는 생성 ---
# 보통은 별도 query 파일을 쓰지만, 여기서는 임의 생성
query_vectors = np.random.rand(1000, dim).astype(np.float32)

# --- 인덱스 로드 ---
print("Loading index from disk...")
index = StaticDiskIndex(
    index_directory=index_dir,
    num_threads=num_threads,
    num_nodes_to_cache=num_nodes_to_cache,
)

# --- 검색 성능 측정 ---
neighbors_all = []
distances_all = []

print("Running search queries...")
start = time.time()
for q in query_vectors:
    neighbors, distances = index.search(q, k_neighbors=k, complexity=complexity)
    neighbors_all.append(neighbors)
    distances_all.append(distances)
end = time.time()

# --- 결과 정리 ---
neighbors_all = np.stack(neighbors_all)
distances_all = np.stack(distances_all)

total_time = end - start
avg_latency_ms = (total_time / len(query_vectors)) * 1000

print(f"\nSearch completed.")
print(f"Total time for {len(query_vectors)} queries: {total_time:.4f} seconds")
print(f"Average latency per query: {avg_latency_ms:.2f} ms")
print(f"Top-1 neighbor indices for each query:\n{neighbors_all[:, 0]}")
