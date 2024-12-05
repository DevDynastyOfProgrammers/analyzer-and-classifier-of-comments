# Импортируем необходимые библиотеки
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем предобученную модель и токенайзер
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 метки: спам или нет

model.to(device)

# 1. Загрузка и подготовка данных
# Предположим, CSV файл содержит колонки "text" (текст отзыва) и "label" (1 - спам, 0 - не спам)
df = pd.read_csv('Review.csv')

# Разделяем данные на тренировочные и тестовые
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Преобразуем данные в формат, который понимает Hugging Face Dataset
train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
test_dataset = Dataset.from_dict({'text': test_texts.tolist(), 'label': test_labels.tolist()})


# Функция токенизации текста
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)


# Токенизация тренировочного и тестового набора
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Удаляем ненужные колонки (оставляем только input_ids, attention_mask и label)
train_dataset = train_dataset.remove_columns(['text'])
test_dataset = test_dataset.remove_columns(['text'])

# Преобразуем данные в формат, который понимает модель (torch.tensor)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Проверяем тренировочные данные после токенизации
print(train_dataset[0])


# Преобразуем метки в целые числа
def cast_labels(example):
    example['label'] = int(example['label'])
    return example


# Применяем функцию к тренировочному и тестовому наборам данных
train_dataset = train_dataset.map(cast_labels)
test_dataset = test_dataset.map(cast_labels)

for column in ['input_ids', 'attention_mask', 'label']:
    train_dataset = train_dataset.map(lambda x: {column: x[column].to(device)})

# Добавляем метрики, для оценки работы модели
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# 2. Настройка тренировки модели
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 3. Обучение модели
trainer.train()

# 4. Оценка и предсказания
trainer.evaluate()

# Пример предсказания
# text_example = "Не вкусный Том Ям, здесь больше не буду заказывать"
text_example = 'ывяп вдла 908ф л л л л дфылво дл лвол олв оговно дл д лд лдл '
inputs = tokenizer(text_example, return_tensors="pt", padding=True, truncation=True, max_length=512)

inputs = {key: val.to(device) for key, val in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

label_map = {0: 'не спам', 1: 'спам'}
print(f"Текст: {text_example}\nКлассификация: {label_map[predictions.item()]}")
