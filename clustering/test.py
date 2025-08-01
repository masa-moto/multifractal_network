import networkx as nx 
import numpy as np 
import matplotlib.pyplot as plt 
import random
from clustering import entropy_based_clustering

def random_color(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.randint(10, 255)
    g = random.randint(10, 255)
    b = random.randint(10, 255)
    return f'#{r:02X}{g:02X}{b:02X}'

if __name__ == "__main__":
    # ハンドウイルカのソーシャルネットワークをベンチマークとして採用。
    # グラフデータはhttps://networkrepository.com/soc-dolphins.phpからDLできる。
    # 同ディレクトリに"soc-dolphins.mtx"がある状態を前提とする。
    file_path = "soc-dolphins.mtx"
    g = nx.read_adjlist(file_path, comments="%")
    
    # 自己ループを削除
    g.remove_edges_from(nx.selfloop_edges(g)) 
    
    # ノードラベルを整数型に変換
    if not isinstance(list(g.nodes)[0], int):
        g = nx.convert_node_labels_to_integers(g)
    
    # クラスタリング
    # clustering_cutoff_sizeで指定した数より多いノードを含むクラスタのみを返す
    # GE_thresholdで指定した値よりも、グラフエントロピーの変化が小さくなったらそのクラスタ構造を確定させる
    clusters = entropy_based_clustering(g, cluster_cutoff_size=2, GE_threshold=0)
    num_clusters = len(clusters)
    
    # クラスタリングから弾かれたノードを調査
    validation_clst = set()
    for clst in clusters:
        idx, cluster = clst
        validation_clst |= cluster    
    outlanders = set(g.nodes()) - validation_clst
    
    #1つ以上のクラスタに入っていないノードがある場合、これを表示する
    if outlanders:
        print(f"outlanders:{outlanders}")
    
    fig = plt.figure(dpi = 300)
    ax = fig.add_subplot()
    pos = nx.nx_agraph.graphviz_layout(g, prog = "sfdp")
    nx.draw_networkx_edges(g, pos=pos, ax = ax, label = None, alpha = .5, width = 1)
    
    # 大規模なクラスタを最後に（一番上に）描画するようにクラスタサイズで並び替えてから描画処理
    cc = set()
    max_node_size, min_node_size = 100, 10
    node_size_step = (max_node_size - min_node_size)/(len(clusters))
    for i, clust in enumerate(sorted(clusters, key = lambda x:len(x[1]), reverse=False)):
        idx, cluster = clust
        
        # 色被りを避けるためにwhileでループを回している
        cluster_color = random_color(idx)
        while cluster_color in cc:
            cluster_color = random_color(idx+random.randint(0, idx))
        nx.draw_networkx_nodes(
            g, pos, ax = ax,
            nodelist=list(cluster),
            node_color = cluster_color,
            node_size = max_node_size-idx,  #大規模クラスタの方が小さく、手前側に描画
            alpha = i/len(clusters),        #小規模なクラスタを透過度高めで描画
            label=f"{i}",
            )
        cc |= {cluster_color}
        
    # クラスタに属さないノードがあった場合、黒色で描画
    if outlanders:
        nx.draw_networkx_nodes(g, pos, outlanders, node_color=r"#000000", node_size = 15)
    
    # test.pngに画像を保存する。
    fig.savefig("./test.png")