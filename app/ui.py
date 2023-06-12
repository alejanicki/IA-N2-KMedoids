import numpy as np
import matplotlib.pyplot as plt

# Gerar visualização Scatter para Agrupamento
def show_scatter(data, clusters):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, cluster in enumerate(set(clusters)):
        cluster_points = np.array([data[j] for j, c in enumerate(clusters) if c == cluster])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {cluster}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agrupamento')
    plt.legend()
    plt.show()