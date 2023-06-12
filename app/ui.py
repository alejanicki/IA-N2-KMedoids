import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Gerar visualização Scatter para Agrupamento
def show_scatter(data, clusters):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, cluster in enumerate(set(clusters)):
        cluster_points = np.array(
            [data[j] for j, c in enumerate(clusters) if c == cluster])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[i], label=f'Cluster {cluster}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agrupamento')
    plt.legend()
    plt.show()

# Gerar matriz de confusão para Classificação
def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    classes = np.unique(y_true)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Classe Preditada')
    plt.ylabel('Classe Real')
    plt.show()

# Gerar visualização da linha de saída para Regressão
def show_regression(X, y, y_pred):
    plt.scatter(X, y, color='b', label='Dados reais')
    plt.plot(X, y_pred, color='r', label='Linha de saída')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regressão')
    plt.legend()
    plt.show()