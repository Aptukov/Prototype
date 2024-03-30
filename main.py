# Импортируем нужные библиотеки
import pandas as pd
from summarizer import Summarizer
from transformers import BertTokenizer, BertModel
from bert_score import score

# Загрузка данных
train_data = pd.read_csv('gazeta_train.jsonl',on_bad_lines='skip')
test_data = pd.read_csv('gazeta_test.jsonl',on_bad_lines='skip')
val_data = pd.read_csv('gazeta_val.jsonl',on_bad_lines='skip')

# Инициализация экстрактивного суммаризатора
extractive_summarizer = Summarizer()

# Функция для абстрактивной суммаризации
def abstractive_summarization(text):
    abstractive_summarizer = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    abstractive_summary = abstractive_summarizer.generate(**inputs, max_length=150, min_length=60)
    return abstractive_summary

# Применение экстрактивного суммаризатора, а затем абстрактивного суммаризатора на каждую новость
predicted_summaries = []
for text in data['text']:
    extractive_summary = extractive_summarizer(text, min_length=60, max_length=150)
    abstractive_summary = abstractive_summarization(extractive_summary)
    predicted_summaries.append(abstractive_summary)

# Добавление результатов в новый столбец 'predicted'
data['predicted'] = predicted_summaries

# Создание списка справочных кратких описаний для метрики Bert score
references = data['text'].tolist()

# Проверка работы суммаризатора с помощью метрики Bert score
bert_scores = []
for i in range(len(predicted_summaries)):
    score_result = score(predicted_summaries[i], [references[i]], lang='ru')
    bert_scores.append(score_result)

# Добавление результатов Bert score в датафрейм
data['bert_score'] = bert_scores

# Сохранение результата в новый файл
data.to_csv('news_with_predicted_summaries.csv', index=False)