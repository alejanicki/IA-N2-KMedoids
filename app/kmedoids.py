import numpy as np

# Implementação do algoritmo K-medoids
def k_medoids_clustering(data, num_clusters):
    # Inicialização aleatória dos medoids
    np.random.seed(0)
    medoids_indices = np.random.choice(len(data), num_clusters, replace=False)
    medoids = [data[i] for i in medoids_indices]

    # Atribuição dos pontos aos clusters
    clusters = assign_points_to_clusters(data, medoids)

    # Atualização dos medoids
    new_medoids_indices = update_medoids(data, clusters)

    # Iterações até convergência
    while not np.array_equal(medoids_indices, new_medoids_indices):
        medoids_indices = new_medoids_indices
        medoids = [data[i] for i in medoids_indices]

        clusters = assign_points_to_clusters(data, medoids)

        new_medoids_indices = update_medoids(data, clusters)

    return clusters

# Atribuição dos pontos aos clusters
def assign_points_to_clusters(data, medoids):
    clusters = []
    for point in data:
        distances = [distance(point, medoid) for medoid in medoids]
        cluster_index = np.argmin(distances)
        clusters.append(cluster_index)
    return clusters

# Cálculo da distância entre dois pontos
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Atualização dos medoids
def update_medoids(data, clusters):
    new_medoids_indices = []
    for cluster_index in range(len(set(clusters))):
        cluster_points = [data[i] for i, cluster in enumerate(clusters) if cluster == cluster_index]
        total_distances = [sum(distance(point, other_point) for other_point in cluster_points) for point in cluster_points]
        new_medoid_index = cluster_points[np.argmin(total_distances)]
        new_medoids_indices.append(data.index(new_medoid_index))
    return new_medoids_indices