import networkx as nx 
import numpy as np 
import sympy as sp 
import random as rd
import matplotlib.pyplot as plt 
import itertools as itt
from typing import Dict, Any, Tuple, List
from numba import njit, prange
import json 




"""
version
    1.0.0: 2025/02/11 program file created. trial for triple generator based model, worked.
    1.0.1: 2025/02/12 modify added nodes index, create temp_graph in apply_replacement(), and use frozenset() for the keys of node_mapping.
    1.1.0: 2025/02/20 create new method "generalized_dimension" that calculate general dimension of deterministic model.
    1.1.1: 2025/03/03 modified condition in set_generator().
    1.1.2: 2025/03/13 set preset label. if not specified label input, automatically set label based on the current generators.
    1.2.0: 2025/05/14 stochastic model added.
    1.2.1: 2025/05/15 bug fix in apply_replacement() and set_generator(). clear_generator() added.
    1.3.0: 2025/07/28 start using git for tracing edit history, and this is the end of the HARD CORDING history management.

"""

class EdgeReplaceGraph(nx.Graph):
    
    def __init__(self) -> None:
        """Edge replacement graph model.
        Generator-based edge replacement graph model.Initial condition of graph is single edge.
        Use 
            - set_initial_label() to set label on a single edge(initial graph),
            - set_generator() to register generator information,
            - apply_replacement() to execute edge replacement procedure.

        args
        -----------
        no args needed for constructor.
        """
        super().__init__()
        self.replacement_rules: Dict[str, List[Tuple[float, nx.Graph, int, int]]] = {}    #store generator information (label, graphObj, root1, root2)
        
    def set_initial_label(self, label)-> None:
        """initialize label of edge.
        initial condition of graph is a single edge.
        the label of initial condition should be specified corresponding to the label of generator.

        Args:
            label (str): label of generator. label is to correspond to one of generator's. 
        """
        super().add_edge(0, 1, label = label)
    
    def import_graph_data(self, data_path):
        assert nx.number_of_nodes(self) == 0, "graph is not empty. Aborting import."
        with open(data_path, "r") as f:
            data = json.load(f)
        _imported = nx.readwrite.json_graph.node_link_graph(data)
        self.clear()
        self.add_edges_from(_imported.edges(data=True))
        # self.add_nodes_from(_imported.nodes(data=True))
    
    def save_graphdata(self, data_path):
        data = nx.readwrite.json_graph.node_link_data(self, edges = "edges")
        with open(data_path, "w") as f:
            json.dump(data, f)
            
    def clear_generator(self) -> None:
        self.replacement_rules = {} #clear generator information.
        print("all information of generator is cleared.")
        
    def set_generator(
        self,
        label:str = None,
        generator:nx.Graph = None,
        root_node1:int = 0,
        root_node2:int = 1,
        probability:float = 1.0
        ) -> None:
        """register generator information into instance of the class.
        generator information should include folloings:
            - label of generator
            - probability of generator
            - generator structure 
            - the pair of root node
            
        Args:
            label (str): label of generator. edge replacement is executed according to the generator corresponding to the its label
            probability (float): probability of generator. this should be in (0, 1].
            generator (nx.Graph): generator structure. this should be connected, undirected graph including label on each edges.
            root_node1 (int): one of the root node.
            root_node2 (int): the other root node.
        """
        if not (0 < probability <= 1.0):
            raise ValueError("probability must be in (0, 1]")

        if generator is None:
            generator = nx.Graph()
            
        preset_label = f"{len(self.replacement_rules)}"

        if label == None: #if not specified label, label is automatically set according to the number of generators.
            label = preset_label
            
        if label not in self.replacement_rules:
            self.replacement_rules[label] = []
        
        if nx.dijkstra_path_length(generator, root_node1, root_node2) != nx.diameter(generator)\
            or nx.dijkstra_path_length(generator, root_node1, root_node2) == 1:
                print(
                    f"""
                    Warining: the distance between roots is less than the diameter of generator, 
                    which is {nx.dijkstra_path_length(generator, root_node1, root_node2)}. Check the struture.
                    """)
        
        self.replacement_rules[label].append((probability, generator, root_node1, root_node2))    
        self.offset_start = max(2, nx.number_of_nodes(generator))            
        # if label not in self.replacement_rules.keys(): # new label should be used for identification of generator
        #     if type(int(label)) == int: # label not taken. Proceed successfully
        #         self.replacement_rules[label] = (generator, root_node1, root_node2)
        #         if nx.dijkstra_path_length(generator, root_node1, root_node2) != nx.diameter(generator)\
        #             or nx.dijkstra_path_length(generator, root_node1, root_node2) == 1: # condition of roots distance and diameter of generator.
        #             #only warning is raised
        #             print('Though generator is set, the distance of root nodes is less than the diameter of generator. check the structure of generator.')
        #         else: #proceed successfully.
        #             pass
        #     else: # Typeerror in label.   
        #         raise TypeError("label must be consisted of str(int) for the calculation of distributions of symbol.")
        # else: # label is taken. other label should have used. no new generator added.
        #     print('the label is already taken. use any other label', self.replacement_rules.keys())
    
    def _per_edge_replacement(self):
        """
        Executing edge replacement.
        Every time this method is called, all edges are replaced according to the label.
        For every single edge, the generator is chosen according to the probabilities.
        The replaced graph is re-labeled.
        """
        offset = nx.number_of_nodes(self)#+1
        edgenum = nx.number_of_edges(self)
        node_mapping = {}
        edges_buffer = []
        for i, (u, v, data) in enumerate(self.edges(data=True)):
            # print(f'{i+1}/{edgenum}', end = "\r")
            label: str | Any = data.get("label")
            generator_candidates: nx.Graph = self.replacement_rules[label]
            prob = [p for p, *_ in generator_candidates]
            chosen = rd.choices(generator_candidates, weights=prob, k=1)[0]
            generator: nx.Graph
            r1: int
            r2: int
            _, generator, r1, r2 = chosen
            for node in generator.nodes():  #use hashable object. (set is to frozenset what list is to tuple)
                if (node == r1):#root node replacement
                    node_mapping[frozenset([str(u), str(v), int(node)])] = u
                elif (node == r2):#root node replacement
                    node_mapping[frozenset([str(u), str(v), int(node)])] = v
                else:#non-root node replacement
                    # new_graph.add_node(node + last_node_id)
                    node_mapping[frozenset([str(u), str(v), int(node)])] = node+offset

            for edge in generator.edges(data=True):
                edges_buffer.append(
                    (
                    node_mapping[frozenset([str(u), str(v), int(edge[0])])],
                    node_mapping[frozenset([str(u), str(v), int(edge[1])])],
                    edge[2]
                    )
                )
                if u == v:
                    print(f"[!] self-loop in base graph at edge ({u}, {v})")
            offset += nx.number_of_nodes(generator)
            
        temp_graph = nx.Graph()
        temp_graph.add_edges_from(edges_buffer)
        temp_graph = nx.convert_node_labels_to_integers(temp_graph)
        self.clear()
        self.add_edges_from(temp_graph.edges(data=True))
        
    def _per_step_replacement(self):
        """
        Executing edge replacement.
        Every time this method is called, all edges are replaced according to the label.
        The generator to be replacer is fixed in advance of procedure of edge replacement.
        The replaced graph is re-labeled.
        """
        offset = nx.number_of_nodes(self)#+1
        edgenum = nx.number_of_edges(self)
        node_mapping = {}
        edges_buffer = []
        generator_candidates: nx.Graph = self.replacement_rules[label]
        prob = [p for p, *_ in generator_candidates]
        chosen = rd.choices(generator_candidates, weights=prob, k=1)[0]
        generator: nx.Graph
        r1: int
        r2: int
        _, generator, r1, r2 = chosen
        for i, (u, v, data) in enumerate(self.edges(data=True)):
            # print(f'{i+1}/{edgenum}', end = "\r")
            
            label: str | Any = data.get("label")
            
            for node in generator.nodes():  #use hashable object. (set is to frozenset what list is to tuple)
                if (node == r1):#root node replacement
                    node_mapping[frozenset([str(u), str(v), int(node)])] = u
                elif (node == r2):#root node replacement
                    node_mapping[frozenset([str(u), str(v), int(node)])] = v
                else:#non-root node replacement
                    # new_graph.add_node(node + last_node_id)
                    node_mapping[frozenset([str(u), str(v), int(node)])] = node+offset

            for edge in generator.edges(data=True):
                edges_buffer.append(
                    (
                    node_mapping[frozenset([str(u), str(v), int(edge[0])])],
                    node_mapping[frozenset([str(u), str(v), int(edge[1])])],
                    edge[2]
                    )
                )
                if u == v:
                    print(f"[!] self-loop in base graph at edge ({u}, {v})")
            offset += nx.number_of_nodes(generator)
            
        temp_graph = nx.Graph()
        temp_graph.add_edges_from(edges_buffer)
        temp_graph = nx.convert_node_labels_to_integers(temp_graph)
        self.clear()
        self.add_edges_from(temp_graph.edges(data=True))
    
    def apply_replacement(self, mode = "per_edge", **kwargs):
        self._replacement_modes = {
        "per_edge" : self._per_edge_replacement,
        "per_step" : self._per_step_replacement
        }
        if mode not in self._replacement_modes:
            raise ValueError(f'Unknown mode [{mode}]')
        return self._replacement_modes[mode](**kwargs)
    
    def shuffle_node_labels(self, seed=None, return_mapping = False):
        if seed is not None:
            rd.seed(seed)
        old_nodes = list(self.nodes())
        shuffled_nodes = old_nodes.copy()
        rd.shuffle(shuffled_nodes)
        mapping = dict(zip(old_nodes, shuffled_nodes))
        tmp_graph = nx.relabel_nodes(self, mapping=mapping, copy=True)
        self.clear()
        self.add_nodes_from(tmp_graph.nodes(data=True))
        self.add_edges_from(tmp_graph.edges(data=True))
        assert self
        if return_mapping:
            return mapping
        
    
    def root2symb_distribution(self, label):
        """
        returns distributions of the symbol on the path between root nodes in specific generator.
        the number of distributions depends on the number of path between the roots.
        
        Args:
            self    : self
            label   : the label of generator which you get the distribution of symbol 
        """
        symbol_distribution = []
        generator = self.replacement_rules[label][0]
        r1 = self.replacement_rules[label][1]
        r2 = self.replacement_rules[label][2]
        paths = nx.all_shortest_paths(generator, r1, r2)
        print("#shortest path:", len(paths))
        for p in paths:
            distribution = []
            for i in range(len(p))-1:
                e1, e2 = p[i], p[i+1]
                distribution[int(generator[e1][e2]['label'])] += 1
            symbol_distribution.append(distribution)
        return symbol_distribution
    
    def graph2symb_distribution(self, label):
        symbol_distribution = []
        generator = self.replacement_rules[label][0]
        for e in generator.edges(data=True):
            e1, e2, l = e
            symbol_distribution[int(generator[e1][e2]['label'])] += 1
        return symbol_distribution
    
    def all_generator2symb(self):
        all_symbols = []
        for label in range(len(self.replacement_rules)):
            generator = self.replacement_rules[label][0]
            symbols = []
            for e in generator.edges(data = True):
                e1, e2, l = e
                symbols[int(generator[e1][e2]['label'])] += 1
            all_symbols.append(symbols)
        return all_symbols
               
    def spectral_radius(self, matrix):
        """ 
        returns spectral radius of input.
        """
        return max(abs(np.linalg.eigvals(matrix)))
    
    def min_spectral_radius(self, all_distributions):
        """generate matrix (or list in list) using vectors

        Args:
            vectors (_type_): _description_
        """
        iterator = itt.product(*all_distributions)
        spec_rad_list = []
        for combo in iterator:
            spec_rad_list.append(self.spectral_radius(combo))
        return min(spec_rad_list)
    
    def partition_sum(self, q, t):
        #ジェネレータの置換ルールとエッジの本数から重みに関する行列を計算する箇所
        symbols = self.all_generator2symb()
        np_mtx = np.array(symbols)
        sp_mtx = sp.Matrix(np_mtx)
        
        #行列の生成自体はsympyを用いて行い、値を返すときにsubsとか使って代入する。
        pass
    
    def diameter_calc(self, t):
        # 各ジェネレータの置換基点対間のパスから各記号の出現回数を集め、それらの直積からなる行列集合を構成する箇所
        
        # 構成した行列集合のスペクトル半径を計算する箇所
        # 最小スペクトル半径を特定し、そのt乗でグラフ全体の直径の推定を与える箇所->return
        pass
    
    def generalized_dimension(self, q_range=(-100, 100), q_bins = None,  fig_path = None, eps = 1e-10):
        """calculate generalized dimension of model.
        Args:
            q_range (tuple): min and max value of parameter q, default to -100 to 100.
            fig_path (str, None): path of output including extension(eg. .png .jpg). default to None, which implies there's no figure output. see also: https://docs.python.org/ja/3/library/stdtypes.html#truth-value-testing
        """
        if q_bins ==None:
            q_bins = (max(q_range) - min(q_range))*2
        q_arr = np.linspace(q_range[0], q_range[1], q_bins)
        # gather label information of roots paths for all generator
        symbols_list = []
        #直径と分割和から一般化次元を計算する箇所
            #各qについて計算を行い、適当な配列に結果を格納する
                #while文を用いて誤差未満になるまで計算を繰り返す箇所
                #tを大きくして誤差を小さくしていく
        #指定されたpathにプロットした画像を出力する箇所
        
        return None


#-------------------------------------------------------------------------------------------------#
@njit
def sample_discrete(p):
    r = np.random.rand()
    acc = 0.0
    for i in range(len(p)):
        acc += p[i]
        if r < acc:
            return i
    return len(p) - 1  # fallback

@njit(parallel = True)
def matrix_lyapunov(M_array, probs, v0, n_steps:int, trials:int = 100):
    """calculate lyapunov exponent referred in Neroli.Zyu's paper.
    Args:
        M (List): List of matrix
        p (List): List of probability corresponding to the Matrix in M
        a (np.array): vector that represents initial condition.
        n (int): length of matrix product. Suppose not to be too big to avoid overflow.
        trial (int, optional): averaging parameter. Defaults to 10.
    Returns:
        float: lyapunov exponent of matrix products.
    """
    log_norms = np.empty(trials)
    for t in prange(trials):
        v = v0.copy()
        s = 0.0
        for _ in range(n_steps):
            idx = sample_discrete(probs)
            v = M_array[idx] @ v
            norm = np.linalg.norm(v)
            s += np.log(norm)
            v = v/norm
        log_norms[t] = s
    return np.mean(log_norms) / n_steps
#-------------------------------------------------------------------------------------------------#
