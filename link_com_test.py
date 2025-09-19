import networkx as nx
import matplotlib.pyplot as plt
from clustering.link_clustering import HLC
from model.edge_replace_model import EdgeReplaceGraph
import numpy as np
from scipy.cluster.hierarchy import dendrogram
# -----------------------------
# 1. グラフ準備（例: Karate Club）
# -----------------------------
# G = nx.karate_club_graph()
#--------------------------
# edge replace model

replace_iteration = 3
c4 = nx.Graph()
c4.add_edges_from([
    [0, 1,{"label": "A"}],
    [1, 2,{"label": "B"}],
    [2, 3,{"label": "A"}],
    [3, 0,{"label": "B"}]
])
c6 = nx.Graph()
c6.add_edges_from([
    [0, 1,{"label": "A"}],
    [1, 2,{"label": "B"}],
    [2, 3,{"label": "A"}],
    [3, 4,{"label": "B"}],
    [4, 5,{"label": "A"}],
    [5, 0,{"label": "B"}]
])

G = EdgeReplaceGraph()
G.set_generator("A", c4, 0, 2, 1)
G.set_generator("B", c6, 0, 3, 1)
# graph.set_generator("C", c8, 0, 4, 1)

G.set_initial_label("A")
# graph.set_generator("B", c6, 0, 3, 1)
# graph.set_generator("C", c8, 0, 4, 1)

G.set_initial_label("A")
for _ in range(replace_iteration):
    G.apply_replacement()
#--------------------------
# # dolphines netwrok
# file_path = "soc-dolphins.mtx"
# G = nx.read_adjlist(file_path, comments="%")

#--------------------------
G.remove_edges_from(nx.selfloop_edges(G))
#--------------------------

print(f"quick graph info: #edges: {len(G.edges())}, #nodes:{len(G.nodes())}, diam:{nx.diameter(G)}")
# 無向グラフ用のadj, edges作成
adj = {n: set(G.neighbors(n)) for n in G.nodes()}
edges = {tuple(sorted(e)) for e in G.edges()}

# -----------------------------
# 2. Link Community クラスタリング
# -----------------------------
hlc = HLC(adj, edges)
threshold = 0.15 #or None
if threshold:
    best_P, list_D = hlc.single_linkage(threshold=threshold)
    
else:
    best_P, best_S, best_D, list_D = hlc.single_linkage()


# -----------------------------
# 3. エッジコミュニティ -> ノードコミュニティ変換
# -----------------------------
cid2nodes = {}
for edge, cid in best_P.items():
    cid2nodes.setdefault(cid, set()).update(edge)
if threshold:
    print(f"threshold = {threshold:.4f},  num culster={len(cid2nodes.values())}")
else:
    print(f"Partition density D_max = {best_D:.4f}, threshold S_max = {best_S:.4f},  num culster={len(cid2nodes.values())}")

# 色リストを用意（コミュニティ数に応じて拡張可）
colors = plt.cm.tab20.colors  # 最大20色
cid2color = {cid: colors[i % len(colors)] for i, cid in enumerate(cid2nodes)}

# -----------------------------
# 4. 可視化
# -----------------------------
pos = nx.nx_agraph.pygraphviz_layout(G, prog="sfdp")
plt.figure(figsize=(8, 6))

# コミュニティごとにエッジを描画
for cid, nodes in cid2nodes.items():
    sub_edges = [e for e in G.edges() if set(e) <= nodes]
    nx.draw_networkx_edges(G, pos, edgelist=sub_edges, edge_color=[cid2color[cid]]*len(sub_edges), width=3)

# ノードは全体を描画（境界色を黒で統一）
nx.draw_networkx_nodes(G, pos, node_size=10, node_color='lightgray', edgecolors='black')
# nx.draw_networkx_labels(G, pos)

plt.title("Link Communities (Edge-based, overlapping)")
plt.axis('off')
plt.savefig("link_communities.png", dpi=300)

#--------------------
# dendrogram 可視化
#--------------------
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# HLC 実行
hlc = HLC(adj, edges)
if threshold:
    best_P, best_S, best_D, list_D, orig_cid2edge, linkage = hlc.single_linkage(dendro_flag=True)
else:
    best_P, best_S, best_D, list_D, orig_cid2edge, linkage = hlc.single_linkage(dendro_flag=True)

# linkage を scipy 形式に変換
#   scipy 形式: (idx1, idx2, dist, sample_count)
#   idxは0..n-1のleaf番号
#   distは距離（1-類似度など）
#   sample_countはクラスタのサイズ

Z = []
cid2size = {cid: 1 for cid in orig_cid2edge}  # 初期サイズ=1
next_id = max(cid2size.keys()) + 1

for cid1, cid2, S in linkage:
    size1, size2 = cid2size[cid1], cid2size[cid2]
    newsize = size1 + size2
    Z.append([cid1, cid2, 1-S, newsize])  # dist=1-S としておく
    cid2size[next_id] = newsize
    next_id += 1

Z = np.array(Z)

# dendrogram 可視化
plt.figure(figsize=(8, 4))
dendrogram(Z, labels=list(orig_cid2edge.keys()))
if threshold:
    thresh = 1-threshold  # 距離の値でカットライン
else:
    thresh = 1- best_S
plt.axhline(y=thresh, color='r', linestyle='--', lw=1.5)
plt.xlabel("Edge ID")
plt.ylabel(f"Distance (1 - similarity)\nthreshold:{thresh:.4f}")
plt.savefig("link_dendro.png")

# hlc = HLC(adj, edges)
# best_P, best_S, best_D, list_D, orig_cid2edge, linkage = hlc.single_linkage(dendro_flag=True)

# # SciPy形式に変換
# Z = []
# cluster_sizes = {}

# for i, (cid1, cid2, sim) in enumerate(linkage):
#     dist = 1 - sim  # similarity → distance に変換
#     size1 = cluster_sizes.get(cid1, 1)
#     size2 = cluster_sizes.get(cid2, 1)
#     new_size = size1 + size2
#     cluster_sizes[len(cluster_sizes) + len(best_P) + i] = new_size  # 新クラスタIDのサイズを登録

#     Z.append([cid1, cid2, dist, new_size])
# print(len(Z))
# Z = np.array(Z)
# print(Z.shape)

# # dendrogram を描画
# plt.figure(figsize=(8, 4))
# dendrogram(Z)
# plt.title("Link Communities Dendrogram (approx)")
# plt.savefig("link_dendro.png")