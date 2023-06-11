import openpyxl

# Ler dados do arquivo .xlsx
def read_data_from_excel(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active

    data = []
    for row in sheet.iter_rows(values_only=True):
        data.append(row)

    return data

# Ler dados do arquivo .xlsx
file_path = 'C:\\repos\\atividade-n2\\data.xlsx'
data = read_data_from_excel(file_path)
print(data)
