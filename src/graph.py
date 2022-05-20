from scipy.spatial.distance import cosine
import igraph as ig
import json

"""
Constroi o grafo a partir dos nós e arestas
"""
def construct_graph(nodes: list, edges: list) -> ig.Graph:
    labels = [node['label'] for node in nodes]
    G = ig.Graph()
    G.add_vertices(labels)

    # Adiciona arestas ao graph
    for edge in edges:
        nodeA = edge['source']
        nodeB = edge['target']
        if G.are_connected(nodeA, nodeB):
            id = G.get_eid(nodeA, nodeB)
            G.es[id]['weight'] += 1
        else:
            G.add_edge(nodeA, nodeB, weight=1)
    
    return G

"""
Constroi o grafo a partir dos nós utilizando distancia de cosseno e knn para determinar as arestas
"""
def construct_graph_with_cosine(nodes: list, distance_limiar: int, knn_limiar: int) -> ig.Graph:
    labels = [node['label'] for node in nodes]
    G = ig.Graph()
    G.add_vertices(labels)

    for i, node in enumerate(nodes):
        G.vs[i]['x'] = node['x']
        G.vs[i]['y'] = node['y']

    # Para cada constituição
    for nodeA in nodes:
        # Obtem localização
        vectorA = [nodeA['y'], nodeA['x']]

        # Vetor de similaridade da constituição atual
        similarities = []

        # Verifica com todas as constituições
        for nodeB in nodes:
            if str(nodeA['label']) == str(nodeB['label']):
                break
            vectorB = [nodeB['y'], nodeB['x']]

            # Calcula a similaridade entre nodeA e nodeB
            distance_AB = cosine(vectorA, vectorB)

            # Se for maior que o limiar, adiciona a aresta a lista
            if distance_limiar <= distance_AB:
                similarities.append({
                    'distance': distance_AB,
                    'nodeA': nodeA['label'],
                    'nodeB': nodeB['label']
                })
        
        # Obtem as maiores similaridades
        similarities = sorted(
            similarities,
            key=lambda x: x['distance'],
            reverse=True
        )[0:knn_limiar]

        # Adiciona arestas ao graph
        for similarity in similarities:
            if G.are_connected(similarity['nodeA'], similarity['nodeB']):
                id = G.get_eid(similarity['nodeA'], similarity['nodeB'])
                G.es[id]['weight'] += 1
            else:
                G.add_edge(similarity['nodeA'], similarity['nodeB'], weight=1)
    
    # Return graph
    return G

"""
Constroi um json a partir de um graph
"""
def extract_graph(graph: ig.Graph):
    nodes = []
    for i, node in enumerate(graph.vs['name']):
        nodes.append({
            "id": i,
            "cluster": graph.vs[i]['cluster'],
            "label": node
        })

    return nodes