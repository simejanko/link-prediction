import networkx as nx
import numpy as np
#import networkx_addon.similarity.katz as nxk
import community
from collections import Counter
from node2vec import node2vec
from gensim.models import Word2Vec
import scipy.spatial.distance

def rooted_pagerank(G, nbunch, beta=0.85):
    A = nx.to_numpy_matrix(G)
    D = np.diagflat(A.sum(axis=1))

    N = np.linalg.pinv(D).dot(A)
    RPR = (1-beta) * np.linalg.pinv((np.eye(N.shape[0]) - beta * N))
    inv_idx = {n:i for i,n in enumerate(G.nodes())}

    for i, j in nbunch:
        if type(G) is nx.DiGraph:
            yield (i, j, RPR[inv_idx[i], inv_idx[j]])
        else:
            yield (i, j, RPR[inv_idx[i], inv_idx[j]] + RPR[inv_idx[j], inv_idx[i]])

#def katz_similarity(G, nbunch):
#    katz = nxk.katz(G)
#    inv_idx = {n: i for i, n in enumerate(G.nodes())}

#    return ((i, j, katz[inv_idx[i], inv_idx[j]]) for i, j in nbunch)

def common_neighbors(G, nbunch):
    return ((i,j, len(list(nx.common_neighbors(G, i, j)))) for i,j in nbunch)

def common_neighbors_in_in(G, nbunch):
    return ((i,j,len(set(G.predecessors(i)).intersection(G.predecessors(j))) ) for i,j in nbunch)

def common_neigbors_in_out(G, nbunch):
    return ((i, j, len(set(G.predecessors(i)).intersection(G.successors(j)))) for i, j in nbunch)

def common_neigbors_out_in(G, nbunch):
    return ((i, j, len(set(G.successors(i)).intersection(G.predecessors(j)))) for i, j in nbunch)

def common_neihbors_out_out(G, nbunch):
    return ((i, j, len(set(G.successors(i)).intersection(G.successors(j)))) for i, j in nbunch)

def preferential_attachement_in_in(G,nbunch):
    return ((i,j, len(G.predecessors(i)) * len(G.predecessors(j))) for i,j in nbunch)

def preferential_attachement_in_out(G,nbunch):
    return ((i,j, len(G.predecessors(i)) * len(G.successors(j))) for i,j in nbunch)

def preferential_attachement_out_in(G,nbunch):
    return ((i,j, len(G.successors(i)) * len(G.predecessors(j))) for i,j in nbunch)

def preferential_attachement_out_out(G,nbunch):
    return ((i,j, len(G.successors(i)) * len(G.successors(j))) for i,j in nbunch)

def shortest_path_length(G, nbunch):
    #if there is no path between 2 node use d as an estimate. Computing diameter is too expensive.
    d = 14
    for i, j in nbunch:
        try:
            yield (i, j, nx.shortest_path_length(G, i, j))
        except nx.NetworkXNoPath:
            yield (i,j, d)

def reverse_shortest_path_length(G, nbunch):
    nbunch = ((j,i) for i,j in nbunch)
    return shortest_path_length(G, nbunch)

def opposite_friends(G, nbunch):
    return ((i,j, int(i in G[j])) for i,j in nbunch)

def clustering_coefficient_index(G, nbunch):
    nodes = set()
    for i,j in nbunch:
        nodes.add(i)
        nodes.add(j)
    clustering = nx.clustering(G, nodes=nodes)
    return ((i, j, clustering[i] * clustering[j]) for i, j in nbunch)

#http://homes.cs.washington.edu/~fire/pdf/link_tists.pdf
def friends_measure(G, nbunch):
    for i,j in nbunch:
        fm = 0
        for n_i in G[i]:
            for n_j in G[j]:
                if n_i==n_j or n_j in G[n_i]:
                    fm += 1
        yield (i,j,fm)

def louvain_index(G, ebunch):
    partition = community.best_partition(G)
    nx.set_node_attributes(G, 'c', partition)
    com_sizes = Counter(partition.values())
    com_index = dict()
    for i,j in G.edges():
        if partition[i] not in com_index:
            com_index[partition[i]] = 0
        if partition[i] == partition[j]:
            com_index[partition[i]] += 1
    com_index = {c: com_index[c]/(com_sizes[c] * (com_sizes[c]-1)/2) for c in com_index}

    return ((i, j, com_index[partition[i]] if partition[i]==partition[j] else 0) for i,j in ebunch)

def node2vec_features(G, ebunch, p=1, q=1, num_walks=10, walk_length=80, dimensions=128, window_size=10, workers=2, iter=1):
    G = node2vec.Graph(G, type(G) is nx.DiGraph, p, q)
    #G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=iter)

    #COSINE
    #return ((i,j, scipy.spatial.distance.cosine(model.wv[str(i)], model.wv[str(j)])) for i,j in ebunch)


    #MULTIPLY
    return ((i, j, (model.wv[str(i)] * model.wv[str(j)]).tolist()) for i, j in ebunch)