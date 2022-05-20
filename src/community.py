from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import igraph as ig
import pandas as pd

"""
Detecta as comunidades em um grafo por fastgreedy
"""
def community_detection_fast_greedy(graph: ig.Graph, n_clustering: int, optimal_count=False) -> tuple:
    fg = graph.community_fastgreedy()
    ideal = fg.optimal_count

    # Verifica se deve aplicar a quantidade ideal de clustering sugerido pelo fast greedy
    if optimal_count:
        n_clustering = ideal
    
    vertex_clustering = fg.as_clustering(n=n_clustering)
    subgraphs = vertex_clustering.subgraphs()
    graph_aux = graph
    
    # Seta o cluster de cada graph
    for i, constitute in enumerate(graph.vs['name']):
        for j, cluster in enumerate(subgraphs):
            if constitute in cluster.vs['name']:
                graph_aux.vs[i]['cluster'] = j
                break

    # Retorna graph com clusters e quantidade ideal de cluster
    return (graph_aux, ideal)

"""
Detecta as comunidades em um grafo por KMeans
"""
def community_detection_kmeans(nodes: list, n_cluster: int):
    labels = [node['label'] for node in nodes]
    cols = ["Principal 1", "Principal 2"]
    df = pd.DataFrame(columns=cols)
    for i, node in enumerate(nodes):
        df.loc[i, :] = [node['x'], node['y']]
    df.index = labels

    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit_transform(df.loc[:, cols].values)
    return kmeans.labels_, labels

"""
Calcula a melhor quantidade de comunidades em um grafo por KMeans
""" 
def community_best_clusters_silhouette(min_cluster: int, max_cluster: int, nodes: list) -> int:
    silhouette = []

    # Constroi dataframe
    labels = [node['label'] for node in nodes]
    cols = ["Principal 1", "Principal 2"]
    df = pd.DataFrame(columns=cols)
    for i, node in enumerate(nodes):
        df.loc[i, :] = [node['x'], node['y']]
    df.index = labels

    # Calcula os cluster de min até max
    for cluster in range(min_cluster, max_cluster + 1):
        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit_transform(df.loc[:, cols].values)
        silhouette.append({
            'x': cluster, 
            'y': silhouette_score(
              df.loc[:, cols].values, 
              kmeans.labels_
            )
        })
    
    # Obtem o melhor cluster atraves do metodo da silhueta
    best_cluster = sorted(silhouette, key=lambda x: x['y'], reverse=True)[0]['y']
    
    return best_cluster