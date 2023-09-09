import os
import pandas as pd
from django.shortcuts import render
from django.http import FileResponse
from django.conf import settings
from openpyxl import Workbook
import pandas as pd
import numpy as np
import pymorphy2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from django.http import FileResponse, HttpResponseNotFound
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from navec import Navec
# Здесь ваш код нейронной сети, который был указан выше,
# вынесен в отдельную функцию для удобства использования.

# def process_dataframe(df, file_name):

#     nltk.download('punkt')
#     nltk.download('stopwords')
#     df = pd.read_csv('dataRed.csv', sep=';',encoding='utf-8')
#     # file_path = os.path.join(os.path.dirname(__file__), 'static', 'dataRed.csv')
#     # df = pd.read_csv(file_path, sep=';', encoding='utf-8')
#     df_test = pd.read_csv(file_name, encoding='utf-8')

#     df = df.drop(columns=["specialization(специализация)", "id"])
#     df_test['requirements'] = np.nan
#     df_test['terms'] = np.nan
#     df_test['notes'] = np.nan

#     df = df.rename(columns={'responsibilities(Должностные обязанности)': 'responsibilities'})
#     df = df.rename(columns={'requirements(Требования к соискателю)': 'requirements'})
#     df = df.rename(columns={'terms(Условия)'	: 'terms'})
#     df = df.rename(columns={'notes(Примечания)': 'notes'})
#     df = df.dropna(how='all', axis=1)


#     df['requirements'].fillna('', inplace=True)


#     df['terms'].fillna('', inplace=True)


#     df['notes'].fillna('', inplace=True)

#     random_state = 42
#     max_review_len = 100
#     vector_size = 300

#     def preprocess(text, stop_words, punctuation_marks, morph):
#         tokens = word_tokenize(text.lower())
#         preprocessed_text = []
#         for token in tokens:
#             if token not in punctuation_marks:
#                 lemma = morph.parse(token)[0].normal_form
#                 if lemma not in stop_words:
#                     preprocessed_text.append(lemma)
#         return preprocessed_text

#     punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»', ';', '–', '--']
#     stop_words = stopwords.words("russian")
#     morph = pymorphy2.MorphAnalyzer()

#     df['Preprocessed_reqr'] = df.apply(lambda row: preprocess(row['requirements'], punctuation_marks, stop_words, morph), axis=1)
#     df['Preprocessed_terms'] = df.apply(lambda row: preprocess(row['terms'], punctuation_marks, stop_words, morph), axis=1)
#     df['Preprocessed_notes'] = df.apply(lambda row: preprocess(row['notes'], punctuation_marks, stop_words, morph), axis=1)
#     df['Preprocessed_resps'] = df.apply(lambda row: preprocess(row['responsibilities'], punctuation_marks, stop_words, morph), axis=1)

#     navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

#     def vectorize_text_avg(txt, navec):
#         unk = navec['<unk>']
#         text_embeddings = []
#         for token in txt:
#             embedding = navec.get(token, unk)
#             text_embeddings.append(embedding)
    
#         if text_embeddings: # check if list is not empty
#             avg_vector = np.mean(text_embeddings, axis=0)
#         else: 
#             avg_vector = np.zeros(vector_size) # if text was empty, return zero vector
#         return avg_vector

#     df['req_vec'] = df.apply(lambda row: vectorize_text_avg(row['Preprocessed_reqr'], navec), axis=1)
#     df['terms_vec'] = df.apply(lambda row: vectorize_text_avg(row['Preprocessed_terms'], navec), axis=1)
#     df['notes_vec'] = df.apply(lambda row: vectorize_text_avg(row['Preprocessed_notes'], navec), axis=1)
#     df['resp_vec'] = df.apply(lambda row: vectorize_text_avg(row['Preprocessed_resps'], navec), axis=1)

#     df = df.drop(columns=["Preprocessed_reqr", "Preprocessed_terms", "Preprocessed_notes"])

#     df_test['responsibilities'].fillna('', inplace=True)
#     df_test['requirements'].fillna('', inplace=True)
#     df_test['terms'].fillna('', inplace=True)
#     df_test['notes'].fillna('', inplace=True)

#     df_test['Preprocessed_reqr'] = df_test.apply(lambda row: preprocess(row['requirements'], punctuation_marks, stop_words, morph), axis=1)
#     df_test['Preprocessed_terms'] = df_test.apply(lambda row: preprocess(row['terms'], punctuation_marks, stop_words, morph), axis=1)
#     df_test['Preprocessed_notes'] = df_test.apply(lambda row: preprocess(row['notes'], punctuation_marks, stop_words, morph), axis=1)
#     df_test['Preprocessed_resps'] = df_test.apply(lambda row: preprocess(row['responsibilities'], punctuation_marks, stop_words, morph), axis=1)

#     df_test['req_vec'] = df_test.apply(lambda row: vectorize_text_avg(row['Preprocessed_reqr'], navec), axis=1)
#     df_test['terms_vec'] = df_test.apply(lambda row: vectorize_text_avg(row['Preprocessed_terms'], navec), axis=1)
#     df_test['notes_vec'] = df_test.apply(lambda row: vectorize_text_avg(row['Preprocessed_notes'], navec), axis=1)
#     df_test['resp_vec'] = df_test.apply(lambda row: vectorize_text_avg(row['Preprocessed_resps'], navec), axis=1)

#     df_test = df_test.drop(columns=["Preprocessed_reqr", "Preprocessed_terms", "Preprocessed_notes"])

#     from sklearn.multiclass import OneVsRestClassifier
#     from sklearn.preprocessing import MultiLabelBinarizer

#     df_train = pd.DataFrame(columns=['vector', 'label'])

# # добавляем вектора и метки
#     for index, row in df.iterrows():
#         df_train = df_train.append({'vector': row['req_vec'], 'label': ['requirements']}, ignore_index=True)
#         df_train = df_train.append({'vector': row['terms_vec'], 'label': ['terms']}, ignore_index=True)
#         df_train = df_train.append({'vector': row['notes_vec'], 'label': ['notes']}, ignore_index=True)

# # преобразуем вектора в матрицу признаков
#     X = np.stack(df_train['vector'].to_numpy())

# # преобразуем метки в бинарную матрицу
#     mlb = MultiLabelBinarizer()
#     y = mlb.fit_transform(df_train['label'])

# # делим на обучающую и тестовую выборку
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# # обучаем модель
#     model = OneVsRestClassifier(LogisticRegression(random_state=random_state))
#     model.fit(X_train, y_train)

# # теперь можно предсказывать категорию для векторов из df_test
#     y_pred = model.predict(np.stack(df_test['resp_vec'].to_numpy()))
#     predicted_labels = mlb.inverse_transform(y_pred)


#     df_test['predicted_labels'] = predicted_labels
#     for index, row in df_test.iterrows():
#         if 'requirements' in row['predicted_labels']:
#             df_test.loc[index, 'requirements'] = row['responsibilities']
#         if 'terms' in row['predicted_labels']:
#             df_test.loc[index, 'terms'] = row['responsibilities']
#         if 'notes' in row['predicted_labels']:
#             df_test.loc[index, 'notes'] = row['responsibilities']

#     df_pred = df_test.drop(columns=["Preprocessed_resps", "req_vec", "terms_vec", "notes_vec", "resp_vec", "predicted_labels"])

#     # df_pred['requirements'].to_csv("requirements.csv", index=False)
#     # df_pred['terms'].to_csv("terms.csv", index=False)
#     # df_pred['notes'].to_csv("notes.csv", index=False)
#     df_pred.to_csv("newData.csv", index=False)
    

#     return df_pred


import pandas as pd
import csv
from io import StringIO

from django.http import HttpResponse
from django.template import loader



def process_text_to_csv(text):
    # Разделители для CSV
    delimiter = ','

    # Разбиваем текст на строки
    text_lines = text.split('\n')

    # Создаем список для хранения обработанных строк
    csv_data = []

    # Загружаем категории и слова из category_words.csv
    categories_and_words = load_categories_and_words()

    # Создаем множество для ускорения проверки категории слова
    category_set = {category for category, _ in categories_and_words}

    # Обрабатываем каждую строку
    for line in text_lines:
        # Заменяем символ ";" на другой разделитель, чтобы избежать конфликтов
        line = line.replace(";", "|")

        # Разбиваем строку на слова
        words = line.split()

        # Создаем список для хранения обработанных слов
        processed_words = []

        # Обрабатываем каждое слово
        for word in words:
            word_category = get_word_category(word, categories_and_words)
                
            if word_category == 'C':
                # Слово находится в категории "C", выделяем красным
                processed_word = f'<span style="color: red;">{word}</span>'
            
            elif word_category in category_set:
                # Слово находится в одной из категорий A, AA, AAA, B, BB, BBB, выделяем зеленым
                processed_word = f'<span style="color: green;">{word}</span>'
            else:
                # Если в слове совпадает 5 или более букв с категорией, также выделяем соответствующим цветом
                for category, words in categories_and_words:
                    for w in words:
                        if len(w) >= 5 and w[:5] == word[:5]:
                            if category == 'C':
                                processed_word = f'<span style="color: red;">{word}</span>'
                            elif category in category_set:
                                processed_word = f'<span style="color: green;">{word}</span>'
                            break
                    else:
                        continue
                    break
                else:
                    processed_word = word

            processed_words.append(processed_word)

        # Объединяем обработанные слова в строку
        processed_line = ' '.join(processed_words)

        # Добавляем обработанную строку в список
        csv_data.append(processed_line)

    # Объединяем обработанные строки в одну строку с разделителями
    csv_text = delimiter.join(csv_data)

    return csv_text

def load_categories_and_words():
    categories_and_words = []
    try:
        with open('category_words.csv', 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            next(csv_reader)  # Пропускаем первую строку (заголовок)
            for row in csv_reader:
                if len(row) > 1:
                    category = row[0].strip()
                    words = row[1].split(', ')
                    categories_and_words.append((category, words))
    except FileNotFoundError:
        pass  # Обработка случая, когда файл category_words.csv не существует
    return categories_and_words

def get_word_category(word, categories_and_words):
    for category, words in categories_and_words:
        if word in words:
            return category
    return None



import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('processed_data.csv', encoding='utf-8')

X = data['pr_txt']
y = [(category, rating) for category, rating in zip(data['Категория'], data['Уровень рейтинга'])]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=10000) 
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


loaded_model = joblib.load('ridge_model.pkl')
loaded_model2 = joblib.load('ridge_model_2.pkl')



def index(request):
    context = {}
    processed_csv = ''  

    if request.method == 'POST':
        file = request.FILES.get('file_to_upload')
        user_input_text = request.POST.get('commentarea', '')

        if file:
            file_name = file.name
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            if file_name.endswith('.txt'):
                # Считывание текста из файла
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    input_text = txt_file.read()

                input_text_tfidf = tfidf_vectorizer.transform([input_text])

                # Предсказание категории и рейтинга
                predicted_category = loaded_model.predict(input_text_tfidf)
                predicted_rating = loaded_model2.predict(input_text_tfidf)


                processed_csv = process_text_to_csv(input_text)

                from bs4 import BeautifulSoup


                soup = BeautifulSoup(processed_csv, 'html.parser')
                red_word_count = len(soup.find_all('span', style='color: red;'))
                print(red_word_count)

                # Если больше четырех слов подчеркнуты красным, то установить уровень рейтинга и категорию "C"
                if red_word_count > 4:
                    predicted_rating[0] = 'C'
                    predicted_category[0] = 'C'
                

                # Определение новых значений категории на основе рейтинга
                rating_to_category_mapping = {
                    'AA+': 'AA',
                    'AA-': 'AA',
                    'BB+': 'BB',
                    'BB-': 'BB',
                    'A+': 'A',
                    'A-': 'A',
                    'BBB+': 'BBB',
                    'BBB-': 'BBB',
                    'C':'C'
                }

                # Замена категории, если она находится в словаре
                if predicted_rating[0] in rating_to_category_mapping:
                    predicted_category[0] = rating_to_category_mapping[predicted_rating[0]]


                context['predicted_category'] = predicted_category[0]
                context['predicted_rating'] = predicted_rating[0]

                if (predicted_category[0] == 'A' and predicted_rating[0] == 'AAA') or (predicted_category[0] == 'AAA' and predicted_rating[0] == 'A'):
                    context['predicted_category'] = 'AA'
                    context['predicted_rating'] = 'AA'
                
                

            elif user_input_text:
                # Если текст введен вручную, обработайте его и создайте CSV
                processed_csv = process_text_to_csv(user_input_text)

            context['csv_data'] = processed_csv

    return render(request, 'website/index.html', context)


from django.http import FileResponse

def download(request):
    
    if request.session.get('file_processed', False):
        file = open('newData.xlsx', 'rb')
        response = FileResponse(file)
        response['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response['Content-Disposition'] = 'attachment; filename=newData.xlsx'
        
    
        del request.session['file_processed']
        return response

    return HttpResponseNotFound()