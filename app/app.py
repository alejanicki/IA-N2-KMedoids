import openpyxl

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