# Импортируем нужные библиотеки
import pandas as pd
from transformers import BertTokenizer, BertLMHeadModel
from bert_score import score
import json
import torch

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from collections import defaultdict

# Считывание данных из файлов
with open('gazeta_train.jsonl', 'r', encoding='utf-8') as file:
    train = file.readlines()
train_json = [json.loads(line) for line in train]
train_data = pd.DataFrame(train_json)

with open('gazeta_test.jsonl', 'r', encoding='utf-8') as file:
    test = file.readlines()
test_json = [json.loads(line) for line in test]
test_data = pd.DataFrame(test_json)



# Задаём параметр сжатия экстрактивным подходом (20%)
ratio = 0.2

# Функция для экстрактивной суммаризации
def extractive_summarization(text, ratio):
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
    num_sentences = int(len(sentences) * ratio)

    # Извлечение топ предложений
    summary = ' '.join(sentences[idx] for _, idx in ranked_sentences[:num_sentences])

    return summary


# Функция для абстрактивной суммаризации
def abstractive_summarization(text):
    # Загрузим предварительно обученную модель BERT и токенизатор
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertLMHeadModel.from_pretrained('bert-base-uncased',is_decoder=True)

    # Токенизируем и добавляем специальные токены [CLS] and [SEP]
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=150, truncation=True)

    # Получаем выходы модели
    outputs = model.generate(inputs, max_new_tokens=10, min_length=30, length_penalty=1.0, num_beams=4, early_stopping=True)

    # Декодируем сгенерированный текст
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

# Применение экстрактивного суммаризатора, а затем абстрактивного суммаризатора на каждый текст
predicted_summaries = []
for text in train_data['text']:
    extractive_summary = extractive_summarization(text, ratio)
    abstractive_summary = abstractive_summarization(extractive_summary)
    predicted_summaries.append(abstractive_summary)

# Добавление результатов в новый столбец 'predicted'
train_data['predicted'] = predicted_summaries

# Создание списка справочных кратких описаний для метрики Bert score
references = train_data['text'].tolist()

# Проверка работы суммаризатора с помощью метрики Bert score
bert_scores = []
for i in range(len(predicted_summaries)):
    score_result = score(predicted_summaries[i], [references[i]], lang='ru')
    bert_scores.append(score_result)

# Добавление результатов Bert score в датафрейм
data['bert_score'] = bert_scores

# Предполагаем, что у вас уже есть заполненный столбец 'bert_score' в вашем датафрейме data
average_bert_score = data['bert_score'].mean()

print(f'Средний результат метрики Bert score: {average_bert_score}')

# Сохранение результата в новый файл
train_data.to_csv('text_with_predicted', index=False)