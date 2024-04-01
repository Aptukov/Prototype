# Импортируем нужные библиотеки
import pandas as pd
from  summarizer  import  Summarizer

import nltk
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from bert_score import score

from collections import defaultdict
from rouge import Rouge

import json
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Считывание данных из файла
with open('gazeta_train.jsonl', 'r', encoding='utf-8') as file:
    train = file.readlines()
train_json = [json.loads(line) for line in train]
train_data = pd.DataFrame(train_json)

# Удаляем лишние столбцы
del train_data['date']
del train_data['url']
del train_data['title']

# Эмодзи и их текстовые представления
emoji_dict = {
    "🤣": "смешно",
    # В дальнейшем список будет пополняться
}


# Определние тональности сообщений

# Загрузка данных для обучения отдельной модели по определнию тональности
train = pd.read_csv('train.csv')

# Преобразование текста в числовые векторы с помощью TF-IDF
vectorizer = TfidfVectorizer(max_features=100000)
X_train_tfidf = vectorizer.fit_transform(train['text'])
y_train = train['sentiment']

# Обучение модели SVM
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

# Преобразование текста из тестового набора в числовые векторы с помощью TF-IDF
X_test_tfidf = vectorizer.transform(train_data['text'])

# Предсказание эмоциональной категории текста в тестовых данных
sentiments = svm_model.predict(X_test_tfidf)

# Добавление предсказаний в новый столбец в тестовых данных
train_data['sentiment'] = sentiments


# Функция для экстрактивной суммаризации
def extractive_summarization(text):
    sentences = sent_tokenize(text)

    # Подсчет частоты встречаемости слов
    words = [word for sentence in sentences for word in nltk.word_tokenize(sentence.lower()) if word.isalnum()]
    freq = FreqDist(words)

    ranking = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for word in nltk.word_tokenize(sentence.lower()):
            if word in freq:
                ranking[i] += freq[word]

    # Сортировка предложений по рейтингу
    ranked_sentences = sorted(((rank, i) for i, rank in ranking.items()), reverse=True)

    # Вычисление количества предложений для суммаризации
    num_sentences = int(len(sentences) * 0.5)

    # Извлечение топ предложений
    summary = ' '.join(sentences[idx] for _, idx in ranked_sentences[:num_sentences])

    return summary

# Создание модели абстрактивной суммаризации
abstractive_summarization = Summarizer()

# Применение экстрактивного суммаризатора, а затем абстрактивного суммаризатора на каждый текст
predicted_summaries = []
for text in train_data['text']:

    # Преобразование эмодзи в текст
    for emoji_char, text_representation in emoji_dict.items():
        text = text.replace(emoji_char, text_representation)

    # Экстрактивная суммаризация
    extractive_summary = extractive_summarization(text)
    # Абстрактивная суммаризация
    abstractive_summary = abstractive_summarization(text, num_sentences=2)
    result = ''.join(abstractive_summary)

    # Добавление результата в будущий столбец
    predicted_summaries.append(result)

# Добавление результатов в новый столбец 'predicted'
train_data['predicted'] = predicted_summaries

# Создание списка справочных кратких описаний для метрики Bert score
references = train_data['summary'].tolist()

# Проверка работы суммаризатора с помощью метрики Bert score
bert_scores = []
for i in range(len(predicted_summaries)):
    score_result = score([predicted_summaries[i]], [references[i]], lang='ru')
    bert_scores.append(score_result)

# Подсчет метрик для каждой пары значений 'predicted' и 'summary'
bleu_scores = []
mean_rouge_l_scores = []

for index, row in train_data.iterrows():
    predicted_summary = row['predicted']
    reference_summary = row['summary']

    # BLEU score
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_summary.split()], predicted_summary.split(),
                               smoothing_function=smoothing_function)
    bleu_scores.append(bleu_score)

    # Rouge-L score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predicted_summary, reference_summary, avg=True)
    mean_rouge_l_scores.append(rouge_scores['rouge-l']['f'])

# Вычисление среднего значения метрик
mean_bleu = sum(bleu_scores) / len(bleu_scores)
mean_rouge_l = sum(mean_rouge_l_scores) / len(mean_rouge_l_scores)
bert_score = sum([score[1].item() for score in bert_scores]) / len(bert_scores)

# Выводим результаты всех метрик
print("Результат метрики Mean BLEU Score:", mean_bleu) # 0.08
print("Результат метрики Mean Rouge-L Score:", mean_rouge_l) # 0.23
print("Результат метрики Bert score:", bert_score) # 0.76
