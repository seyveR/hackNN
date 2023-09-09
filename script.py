# import pandas as pd

# # Укажите путь к файлу Excel
# excel_file = 'CRA_test.xlsx'

# # Считываем файл Excel
# df = pd.read_excel(excel_file)

# # Укажите имя для файла CSV, в который хотите сохранить данные
# csv_file = 'CRA_test.csv'

# # Сохраняем данные в формате CSV
# df.to_csv(csv_file, index=False)

# print(f'Данные успешно сохранены в файл: {csv_file}')



#22

# import pandas as pd

# df = pd.read_csv('CRA_test.csv', sep=',')

# # Удаление пустых строк, переносов, enter и табуляций из третьего столбца
# df.iloc[:, 2] = df.iloc[:, 2].replace({'\n': ' ', '\t': ' '}, regex=True).str.strip()

# # Сохранение всех трех столбцов и названий столбцов в текстовом файле
# df.to_csv('CRA_test.txt', index=False, sep='\t', line_terminator='\n')

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
with open('CRA_test.txt', 'r', encoding='utf-8') as txt_file:
    # Читаем строки из текстового файла и разделяем их на две колонки
    lines = [line.strip().split('\t') for line in txt_file]

# Открываем CSV файл для записи
with open('CRA_test_new.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Создаем объект для записи CSV
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    # Записываем данные в CSV файл
    csv_writer.writerows(lines)

print("Преобразование завершено.")

