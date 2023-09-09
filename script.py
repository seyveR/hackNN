# import openpyxl

# # Открываем файл XLSX для чтения
# wb = openpyxl.load_workbook('CRA_train_1200.xlsx')

# # Выбираем активный лист (может потребоваться указать конкретный лист по имени)
# sheet = wb.active

# # Открываем текстовый файл для записи
# with open('output.txt', 'w', encoding='utf-8') as txt_file:
#     for row in sheet.iter_rows(values_only=True):
#         # Преобразуем значения в строку и объединяем их с разделителем (например, табуляцией)
#         line = '\t'.join(map(str, row))
#         txt_file.write(line + '\n')

# # Закрываем файлы
# wb.close()


#22

# import pandas as pd

# df = pd.read_csv('output.csv', sep='\t')

# # Удаление пустых строк, переносов, enter и табуляций из третьего столбца
# df.iloc[:, 2] = df.iloc[:, 2].replace({'\n': ' ', '\t': ' '}, regex=True).str.strip()

# # Сохранение всех трех столбцов и названий столбцов в текстовом файле
# df.to_csv('output2.txt', index=False, sep='\t', line_terminator='\n')

# print("Преобразование завершено.")



#33

# import pandas as pd

# df = pd.read_csv('output.csv', sep='\t')

# # Удаление пустых строк, переносов, enter и табуляций из третьего столбца
# df.iloc[:, 2] = df.iloc[:, 2].replace({'\n': ' ', '\t': ' '}, regex=True).str.strip()

# # Создание нового DataFrame, содержащего только второй и третий столбцы
# new_df = df.iloc[:, [1, 2]]

# # Сохранение второго и третьего столбцов и названий столбцов в текстовом файле
# new_df.to_csv('output2.txt', index=False, sep='\t', line_terminator='\n')

# print("Преобразование завершено.")



##44
import csv

# Открываем текстовый файл для чтения
with open('output.txt', 'r', encoding='utf-8') as txt_file:
    # Читаем строки из текстового файла и разделяем их на две колонки
    lines = [line.strip().split('\t') for line in txt_file]

# Открываем CSV файл для записи
with open('outputNEW2.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Создаем объект для записи CSV
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    # Записываем данные в CSV файл
    csv_writer.writerows(lines)

print("Преобразование завершено.")

