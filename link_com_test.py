import networkx as nx
import matplotlib.pyplot as plt
from clustering.link_clustering import HLC

# -----------------------------
# 1. グラフ準備（例: Karate Club）
# -----------------------------
G = nx.karate_club_graph()

# 無向グラフ用のadj, edges作成
adj = {n: set(G.neighbors(n)) for n in G.nodes()}
edges = {tuple(sorted(e)) for e in G.edges()}

# -----------------------------
# 2. Link Community クラスタリング
# -----------------------------
hlc = HLC(adj, edges)
best_P, best_S, best_D, list_D = hlc.single_linkage()

print(f"Partition density D_max = {best_D:.4f}, threshold S_max = {best_S:.4f}")

# -----------------------------
# 3. エッジコミュニティ -> ノードコミュニティ変換
# -----------------------------
cid2nodes = {}
for edge, cid in best_P.items():
    cid2nodes.setdefault(cid, set()).update(edge)

# 色リストを用意（コミュニティ数に応じて拡張可）
colors = plt.cm.tab20.colors  # 最大20色
cid2color = {cid: colors[i % len(colors)] for i, cid in enumerate(cid2nodes)}

# -----------------------------
# 4. 可視化
# -----------------------------
pos = nx.spring_layout(G, seed=42)  # ノード配置固定

plt.figure(figsize=(8, 6))

# コミュニティごとにエッジを描画
for cid, nodes in cid2nodes.items():
    sub_edges = [e for e in G.edges() if set(e) <= nodes]
    nx.draw_networkx_edges(G, pos, edgelist=sub_edges, edge_color=[cid2color[cid]]*len(sub_edges), width=2)

# ノードは全体を描画（境界色を黒で統一）
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgray', edgecolors='black')
nx.draw_networkx_labels(G, pos)

plt.title("Link Communities (Edge-based, overlapping)")
plt.axis('off')
plt.savefig("link_communities.png", dpi=300)
