import networkx as nx
import random
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from features import *
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


NODE2VEC_DIM = 32

def read_graph(file_name, konect=False, directed=False, add_weights=False, remove_loops=True, max_component=True):
    create_using = nx.Graph() if not directed else nx.DiGraph()
    if konect:
        G = read_konect(file_name, create_using=create_using)
    else:
        G = nx.read_edgelist(file_name, nodetype=int, create_using=create_using)

    if max_component:
        if directed:
            G = max(nx.weakly_connected_component_subgraphs(G), key=len)
        else:
            G = max(nx.connected_component_subgraphs(G), key=len)

    if add_weights:
        for edge in G.edges():
            G[edge[0]][edge[1]]['w'] = 1

    if remove_loops:
        G.remove_edges_from(G.selfloop_edges())

    return G

def read_konect(file_name, create_using=nx.Graph()):
    G = create_using
    with open(file_name, newline='') as file:
        reader = csv.reader(file, delimiter=' ')
        #node, node, weight, timestamp
        for i,j,w,t in reader:
            #t = wt.strip().split('\t')[1]
            G.add_edge(int(i), int(j), t=int(t))
    return G

def evaluate(G, test_size, train_size, params,n=10 ,timestamps=False, save_datasets=False, graph_folder=''):
    timestamps_test=timestamps_train=timestamps
    if timestamps == 'mix':
        timestamps_test = True
        timestamps_train = False
    avg_auc = {'lr':0, 'rf':0, 'svm':0}
    avg_coef = None
    for i in range(n):
        #G_c = G.copy()
        G_c = type(G)(G)
        test_pos, test_neg = sample_examples(G_c, test_size, timestamps_test)
        print('Sampled test set.')
        X_test, y_test, _ = construct_features(G_c, test_pos, test_neg, params)
        print('Constructed test features.')

        train_pos, train_neg = sample_examples(G_c, train_size, timestamps_train)
        print('Sampled train set.')
        #print(sum((v==0 for v in nx.degree(G).values())))
        X_train, y_train, feature_names = construct_features(G_c,train_pos, train_neg, params)
        print('Constructed train features.')

        #standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # saving
        if save_datasets:
            np.save('data/npy/X_train_{}.npy'.format(i), X_train)
            np.save('data/npy/X_test_{}.npy'.format(i), X_test)
            np.save('data/npy/y_train_{}.npy'.format(i), y_train)
            np.save('data/npy/y_test_{}.npy'.format(i), y_test)

        #ml
        lr = LogisticRegression(penalty='l1',C=0.01)
        rf = RandomForestClassifier(n_estimators=800, max_depth=13, min_samples_split=50)
        svm = SVC(C=0.1, kernel='rbf', gamma='auto', probability=True, cache_size=1000)

        avg_auc['lr'] += evaluate_ml(lr, X_train, y_train, X_test, y_test)/n
        #avg_auc['rf'] += evaluate_ml(rf, X_train, y_train, X_test, y_test)/n
        #avg_auc['svm'] += evaluate_ml(svm, X_train, y_train, X_test, y_test)/n


        # lr = LogisticRegression(penalty='l2', C=1)
        # lr = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=50)

        # lr = SVC(C=1, kernel='rbf', gamma=0.005, probability=True, cache_size=1500)

        if avg_coef is None:
            avg_coef = lr.coef_/n
        else:
            avg_coef += lr.coef_/n
        print(lr.coef_)
        #plot_lr_weights(lr.coef_, list(feature_names), save=True,show=False, name='{}lg_weights_{}.png'.format(graph_folder, i))
        print('FINISHED {}. ITERATION OF EVALUATION'.format(i+1))
        print()

    print('Average AUC: {}'.format(avg_auc))
    print('Average logistic regression weights:')
    print(avg_coef)
    #plot_lr_weights(avg_coef, list(feature_names), save=True, show=False, name='{}lg_weights_avg.png'.format(graph_folder))
    return avg_auc, avg_coef

def evaluate_ml(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)

def plot_lr_weights(coef, feature_names, save=True, show=False, name='plot.png', agg_node2vec=False):
    if agg_node2vec:
        n2v_pos = coef[0, -NODE2VEC_DIM:] > 0
        pos_sum = np.sum(coef[0, -NODE2VEC_DIM:][n2v_pos])
        neg_sum = np.sum(coef[0, -NODE2VEC_DIM:][~n2v_pos])
        coef[0, -NODE2VEC_DIM] = pos_sum
        coef[0, -NODE2VEC_DIM +1] = neg_sum
        coef = coef[:,:-NODE2VEC_DIM+2]

        feature_names[-NODE2VEC_DIM] = 'node2vec positive weights'
        feature_names[-NODE2VEC_DIM+1] = 'node2vec negative weights'
        del feature_names[-NODE2VEC_DIM+2:]

    colors = cm.get_cmap('RdYlGn')((coef[0]-np.min(coef[0])) / (np.max(coef[0])-np.min(coef[0])))
    labels = [l.split('.')[-1].replace('_', ' ') for l in feature_names]
    plt.barh(range(coef[0].size), coef[0], tick_label=labels, alpha=0.8, color=colors, height=1, align='center')
    plt.xlabel('feature weight')
    plt.title('logistic regression feature weights')
    plt.margins(y=0)
    if show:
        plt.show()
    if save:
        plt.savefig(name, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()


def sample_examples(G, n, timestamps=False):
    neg_s = sample_non_edges(G, n)
    if timestamps:
        pos_s = sorted(G.edges(), key=lambda e:G[e[0]][e[1]]['t'], reverse=True)[:n]
    else:
        pos_s = random.sample(G.edges(), n)

    G.remove_edges_from(pos_s)
    degree = nx.degree(G)
    neg_s, to_remove1 = filter_zero_degree(G, neg_s, degree)
    pos_s, to_remove2 = filter_zero_degree(G, pos_s, degree)
    to_remove = to_remove1.union(to_remove2)
    G.remove_nodes_from(to_remove)
    print('Removed {} nodes (0 degree)'.format(len(to_remove)))
    print('Left with {} positive and {} negative edges.'.format(len(pos_s), len(neg_s)))
    return pos_s, neg_s

def filter_zero_degree(G, nbunch, degree):
    to_remove = set()
    filtered = []
    c = 0
    for i,j in nbunch:
        if degree[i]>0 and degree[j]>0:
            filtered.append((i, j))
        if degree[i] == 0:
            to_remove.add(i)
            c+=1
        if degree[j] == 0:
            to_remove.add(j)
            c+=1

    print('Removing {} edges (0 degree).'.format(c))
    return filtered, to_remove

def sample_non_edges(G, n):
    non_edges = set()
    nodes = G.nodes()
    while len(non_edges) < n:
        n1, n2 = random.choice(nodes), random.choice(nodes)
        if n1 != n2 and n2 not in G[n1]:
            if type(G) is nx.DiGraph:
                non_edges.add((n1,n2))
            else:
                non_edges.add((min(n1,n2), max(n1,n2)))
    return list(non_edges)

def construct_features(G, pos_s, neg_s, params):
    feature_names = []
    undirected_feature_methods = [common_neighbors, nx.jaccard_coefficient, nx.adamic_adar_index,
                                  nx.preferential_attachment, louvain_index, nx.ra_index_soundarajan_hopcroft,
                                  nx.within_inter_cluster, clustering_coefficient_index] #, friends_measure
    common_feature_methods = [shortest_path_length,node2vec_features] #shortest_path_length, rooted_pagerank,
    directed_feature_methods = [opposite_friends, common_neighbors_in_in, common_neigbors_in_out, common_neigbors_out_in,
                                common_neihbors_out_out, reverse_shortest_path_length, preferential_attachement_in_in,
                                preferential_attachement_in_out, preferential_attachement_out_in, preferential_attachement_out_out]

    undirected_feature_methods = [nx.preferential_attachment]
    common_feature_methods = []
    directed_feature_methods = []
    samples = pos_s + neg_s
    X = []
    G_u = G
    if type(G) is nx.DiGraph:
        for method in directed_feature_methods:
            X.append(extract_values(G, samples, method, params))
            feature_names.append(method.__name__)
            print('\tProcessed directed method: {}'.format(method.__name__))

        G_u = G.to_undirected()

    for method in undirected_feature_methods:
        X.append(extract_values(G_u, samples, method, params))
        feature_names.append(method.__name__)
        print('\tProcessed undirected method: {}'.format(method.__name__))

    G_u = None
    for method in common_feature_methods:
        X.append(extract_values(G, samples, method, params))
        feature_names.append(method.__name__)
        print('\tProcessed common method: {}'.format(method.__name__))

    #process multi-features (node2vec)
    atrs = len(X)
    for i in range(atrs):
        if type(X[i][0]) is list:
            for k in range(1, len(X[i][0])):
                X.append([X[i][j][k] for j in range(len(X[i]))])
                feature_names.append('{} {}'.format(feature_names[i], k))
            X[i] = [X[i][j][0] for j in range(len(X[i]))]
            feature_names[i] = '{} {}'.format(feature_names[i], 0)



    y = [1] * len(pos_s) + [0] * len(neg_s)
    return np.array(X).T, np.array(y), feature_names

def extract_values(G, sample, method, params):
    kwargs = {}
    if method in params:
        kwargs = params[method]
    return list(map(lambda x: x[2], method(G, sample, **kwargs)))

def plot_decaying_auc(G, sizes, params, color='b', n=1):
    aucs = []
    #sizes = np.linspace(start_p, end_p, num_points)
    num_edges = G.number_of_edges()
    for s in sizes:
        size = int(round(num_edges * s/2))
        avg_auc, _ = evaluate(G, size, size, params, n=n)
        aucs.append(avg_auc['lr'])

    plt.plot(sizes, aucs, color)


params = {rooted_pagerank: {'beta': 0.85}, nx.ra_index_soundarajan_hopcroft: {'community': 'c'},
          nx.within_inter_cluster: {'community': 'c'},
          node2vec_features: {'num_walks': 5, 'walk_length': 40, 'iter': 1, 'p': 1, 'q': 1,
                              'dimensions': NODE2VEC_DIM, 'window_size': 7}}
sizes =  [0.00005, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4,0.6,0.8]

print('+++++FACEBOOK++++++')
network = read_graph('data/out.facebook-wosn-links', directed=False, konect=True)
test_size = int(round(network.number_of_edges() / 100))
train_size = int(round(network.number_of_edges() / 50))
evaluate(network, test_size, train_size, params, n=3, timestamps=False, save_datasets=False, graph_folder='facebook/')
#plot_decaying_auc(network, sizes, params, color='r')
print('+++++++++++++++')

print('+++++LIVEMOCHA++++++')
network = read_graph('data/out.livemocha', directed=False, konect=False)
test_size = int(round(network.number_of_edges() / 100))
train_size = int(round(network.number_of_edges() / 50))
evaluate(network, test_size, train_size, params, n=3, timestamps=False, save_datasets=False, graph_folder='livemocha/')
#plot_decaying_auc(network, sizes, params)
print('+++++++++++++++')

print('+++++TWITTER++++++')
network = read_graph('data/twitter_combined.txt', directed=True, konect=False)
test_size = int(round(network.number_of_edges() / 100))
train_size = int(round(network.number_of_edges() / 50))
evaluate(network, test_size, train_size, params, n=3, timestamps=False, save_datasets=False, graph_folder='twitter/')
#plot_decaying_auc(network, sizes, params, color='g')
print('+++++++++++++++')

print('+++++DIGG++++++')
network = read_graph('data/out.digg-friends', directed=True, konect=True)
test_size = int(round(network.number_of_edges() / 100))
train_size = int(round(network.number_of_edges() / 50))
evaluate(network, test_size, train_size, params, n=3, timestamps=False, save_datasets=False, graph_folder='digg/')
#plot_decaying_auc(network, sizes, params, color='m')
print('+++++++++++++++')

print('+++++DIGG_TIMESTAMPS++++++')
evaluate(network, test_size, train_size, params, n=1, timestamps=True, save_datasets=False, graph_folder='digg_timestamps/')
print('+++++++++++++++')

print('+++++DIGG_MIXED++++++')
evaluate(network, test_size, train_size, params, n=3, timestamps='mix', save_datasets=False, graph_folder='digg_mix/')
print('+++++++++++++++')

#plot_decaying_auc(network, 0.01, 0.8, params, num_points=5)

""""print('+++++YOUTUBE++++++')
network = read_graph('data/com-youtube.ungraph.txt', directed=False, konect=False)
test_size = int(round(network.number_of_edges() / 100))
train_size = int(round(network.number_of_edges() / 50))
evaluate(network, test_size, train_size, params, n=1, timestamps=False, save_datasets=False, graph_folder='youtube/')
print('+++++++++++++++')"""

#plt.xlabel('portion of removed edges')
#plt.ylabel('AUC')
#plt.show()

