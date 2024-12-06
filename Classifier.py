import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


class Classifier:
    def __init__(
        self,
        model_name="DeepPavlov/rubert-base-cased",
        num_labels=2,
        label_map={0: "Да", 1: "Нет"},
        device=None,
    ):
        """Загрузка стандартного датасета и определение divice"""
        self.label_map = label_map
        # Устройство для вычислений
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("device: " + str(self.device))

        # Загрузка модели и токенизатора
        print("Load tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Load model device...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        print("Load model device: Done")
        self.model.to(self.device)

        self.trainer = None  # Trainer будет создан позже

    def load_data(self, csv_path, test_size=0.2):
        """Загрузка датасета"""
        # Загружаем данные
        print("Load CSV...")
        df = pd.read_csv(csv_path)
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df["text"], df["label"], test_size=test_size
        )

        print("Create dataset...")
        # Преобразуем данные в Dataset
        train_dataset = Dataset.from_dict(
            {"text": train_texts.tolist(), "label": train_labels.tolist()}
        )
        test_dataset = Dataset.from_dict(
            {"text": test_texts.tolist(), "label": test_labels.tolist()}
        )

        # Токенизация
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=512
            )

        print("Tokenize...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Удаляем текстовые колонки
        train_dataset = train_dataset.remove_columns(["text"])
        test_dataset = test_dataset.remove_columns(["text"])

        # Преобразуем метки в целые числа
        train_dataset = train_dataset.map(lambda x: {"label": int(x["label"])})
        test_dataset = test_dataset.map(lambda x: {"label": int(x["label"])})

        # Устанавливаем формат для PyTorch
        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        train_dataset = train_dataset.map(self.cast_labels)
        test_dataset = test_dataset.map(self.cast_labels)

        # Возвращаем датасеты
        return train_dataset, test_dataset

    def cast_labels(self, example):
        example["label"] = int(example["label"])
        return example

    def compute_metrics(self, pred):
        """Метрики"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(
        self,
        train_dataset,
        test_dataset,
        output_dir="./results",
        logging_dir="./logs",
        num_epochs=3,
        batch_size=8,
    ):
        """Обучение модели"""
        # Настройки обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_dir=logging_dir,
        )

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )

        print("Train model...")
        # Запуск обучения
        self.trainer.train()
        print("Train model: Done")

    def evaluate(self):
        if self.trainer is None:
            raise ValueError("Модель должна быть обучена перед оценкой.")
        return self.trainer.evaluate()

    def predict(self, text):
        """Модерация моделью"""
        # Токенизация текста
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        return self.label_map[predictions.item()]

    def save_model(self, save_dir):
        """Сохраняет модель и токенизатор в указанную директорию."""
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Модель и токенизатор сохранены в {save_dir}.")

    def load_model(self, load_dir):
        """Загружает модель и токенизатор из указанной директории."""
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        self.model.to(self.device)
        print(f"Модель и токенизатор загружены из {load_dir}.")


# Пример использования
if __name__ == "__main__":
    classifier = Classifier(label_map={0: "Не спам", 1: "Спам"})

    # 1. Загрузка и подготовка данных
    train_dataset, test_dataset = classifier.load_data("Review.csv")

    # 2. Обучение модели
    classifier.train(train_dataset, test_dataset)

    # 3. Оценка модели
    eval_results = classifier.evaluate()
    print("Результаты оценки:", eval_results)

    # 4. Сохранение модели
    classifier.save_model("./saved_model_spam")

    # 5. Загрузка модели
    classifier.load_model("./saved_model_spam")

    # 6. Проверка текста
    text_example = "ывяп вдла 908ф л л л л дфылво дл лвол олв оговно дл д лд лдл "
    result = classifier.predict(text_example)
    print(f"Текст: {text_example}\nКлассификация: {result}")
