import networkx as nx
import matplotlib.pyplot as plt
from clustering.link_clustering import HLC
from model.edge_replace_model import EdgeReplaceGraph as ERM
# -----------------------------
# 1. グラフ準備（例: c4c6model）
# -----------------------------
# G = nx.karate_club_graph()
G = ERM()
c4 = nx.cycle_graph(4)
nx.convert_node_labels_to_integers(c4, first_label=0)
for i, (u, v, _) in enumerate(c4.edges(data=True)):
    if i % 2 == 0:
        c4.edges[u, v]["label"] = "A" 
    else:
        c4.edges[u, v]["label"] = "B"
c6 = nx.cycle_graph(6)
nx.convert_node_labels_to_integers(c6, first_label=0)

for i, (u, v, _) in enumerate(c6.edges(data=True)):
    if i % 2 == 0:
        c6.edges[u, v]["label"] = "A"
    else:
        c6.edges[u, v]["label"] = "B"

G = ERM()
G.set_generator("A", c4, 0, 2, 1)
G.set_generator("B", c6, 0, 3, 1)
repl_iter = 3

# everytime we apply a replacement, we should initialize the graph to avoid overflowing size of the graph
G.set_initial_label("A")
for i in range(repl_iter):
    G.apply_replacement()


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
# pos = nx.spring_layout(G, seed=42)  # ノード配置固定
pos = nx.nx_agraph.graphviz_layout(G, prog = "sfdp")


plt.figure(figsize=(8, 6))

# コミュニティごとにエッジを描画
for cid, nodes in cid2nodes.items():
    sub_edges = [e for e in G.edges() if set(e) <= nodes]
    nx.draw_networkx_edges(G, pos, edgelist=sub_edges, edge_color=[cid2color[cid]]*len(sub_edges), width=5)

# ノードは全体を描画（境界色を黒で統一）
nx.draw_networkx_nodes(G, pos, node_size=15, alpha = .7)
# nx.draw_networkx_labels(G, pos)

plt.title("Link Communities (Edge-based, overlapping)")
plt.axis('off')
plt.savefig("link_communities.png", dpi=300)
