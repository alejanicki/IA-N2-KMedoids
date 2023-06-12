import openpyxl

from ui import show_scatter, show_confusion_matrix, show_regression
from kmedoids import k_medoids_clustering

# Ler dados do arquivo .xlsx
def read_data_from_excel(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active

    data = []
    for row in sheet.iter_rows(values_only=True):
        data.append(row)

    return data


# Ler dados do arquivo .xlsx
file_path = './data.xlsx'
data = read_data_from_excel(file_path)
print(data)

# Classificação - Exemplo de dados
num_clusters = 3
clusters = k_medoids_clustering(data, num_clusters)
print(clusters)
show_scatter(data, clusters)

y_true = [0, 1, 1, 0, 2, 2]
y_pred = [0, 1, 2, 0, 1, 2]
show_confusion_matrix(y_true, y_pred)

X = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
y_pred = [1, 3, 5, 7, 9]
show_regression(X, y, y_pred)