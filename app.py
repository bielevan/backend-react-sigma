from src.graph  import construct_graph, extract_graph
from src.community import community_detection_fast_greedy
from src.community import community_detection_kmeans
from src.community import community_best_clusters_silhouette
from flask import jsonify, make_response, request
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
app.config.from_object("config.DevConfig")
cors = CORS(app)

@app.route('/fastgreedy', methods=["POST"])
def generate_community_fast_greedy():
    """
    Endpoint para obter clusters via fastgreedy
    """ 
    if request.method == "POST":
        # Obtem dados do body
        request_data = request.get_json()
        nodes = request_data['nodes']
        edges = request_data['edges']
        n_clusters = request_data['n_clusters']
        
        # Constroi o grafo
        graph = construct_graph(
            nodes,
            edges
        )
        # Realiza a detecção de comunidade 
        graph_fg, communities_ideal = community_detection_fast_greedy(
            graph,
            n_clusters
        )

        # Transforma a saida em JSON
        nodes = extract_graph(graph_fg)
        headers = { "Content-Type": "application/json" }
        body = { "nodes": nodes, "communities_ideal": communities_ideal }
        
        return make_response(
            jsonify(body),
            200,
            headers
        )
    
    return make_response(
        "Erro action HTTP",
        400
    )

@app.route('/kmeans', methods=['POST'])
def generate_community_kmeans():
    """
    Endpoint para obter clusters via KMeans
    """
    headers = { 
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    if request.method == "POST":
        # Obtem dados do body
        request_data = request.get_json()  
        nodes = request_data['nodes']
        min_clusters = request_data['min_clusters']
        max_clusters = request_data['max_clusters']

        # Obtem a melhor quantidade de clusters
        best_cluster = community_best_clusters_silhouette(min_clusters, max_clusters, nodes)

        # Clusteriza os nós
        clusters, labels = community_detection_kmeans(nodes, best_cluster)
        message = { 
            "clusters":  [int(elem) for elem in clusters],
            "labels": labels,
            "status": 200,
            "message": "OK"
        }
        response = jsonify(message)
        response.status_code = 200
        return response

    return make_response(
        "Erro action HTTP",
       400,
       headers
    )

@app.route('/silhouette', methods=["POST"])
def best_cluster_by_silhouette():
    """
    Endpoint para obter melhor quantidade de cluster via silhouette
    """
    if request.method == "POST":
        # Obtem dados do body
        request_data = request.get_data()
        min_cluster = request_data['min_cluster']
        max_cluster = request_data['max_cluster']
        nodes       = request_data['nodes']

        # Obtem os clusters
        best = community_best_clusters_silhouette(max_cluster, min_cluster, nodes)
        headers = { "Content-Type": "application/json" }

        return make_response(
            best,
            200,
            headers
        )

    return make_response(
        "Erro action HTTP",
       400 
    )

if __name__ == '__main__':
    app.run(
        port=5000,
        host='0.0.0.0',
        debug=True
    )