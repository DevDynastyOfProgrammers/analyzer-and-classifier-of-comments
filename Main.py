import pandas as pd
import os
from Classifier import Classifier
from DatasetParser import DatasetParser


def learn_spam():
    classifier = Classifier(label_map={0: "Не спам", 1: "Спам"})

    # 1. Загрузка и подготовка данных
    train_dataset, test_dataset = classifier.load_data("ReviewSpam.csv")

    # 2. Обучение модели
    classifier.train(train_dataset, test_dataset)

    # 3. Оценка модели
    eval_results = classifier.evaluate()
    print("Результаты оценки:", eval_results)

    # 4. Сохранение модели
    classifier.save_model("./models/saved_model_spam")


def remove_spam(data_name="ReviewSpam.csv"):
    # 1. Получение модели
    model = Classifier(label_map={0: "Не спам", 1: "Спам"})
    model.load_model("./models/saved_model_spam")

    # 4. Создания датасета с проверенными данными
    parser = DatasetParser()
    parser.parse_for_model_slice(
        data_name, model, "Не спам", "datasets/spam_output.csv"
    )


def learn_moderation():
    classifier = Classifier(label_map={0: "Готов к публикации", 1: "Требует модерации"})

    train_dataset, test_dataset = classifier.load_data("ReviewModer.csv")

    classifier.train(train_dataset, test_dataset)

    eval_results = classifier.evaluate()

    print("Результаты оценки:", eval_results)
    classifier.save_model("./models/saved_model_moderation")


def check_moderation(data_name="ReviewModer.csv"):
    model = Classifier(label_map={0: "Готов к публикации", 1: "Требует модерации"})
    model.load_model("./models/saved_model_moderation")

    parser = DatasetParser()
    parser.parse_for_model_slice(
        data_name, model, "Требует модерации", "datasets/moderation_required_output.csv"
    )
    parser.parse_for_model_slice(
        data_name,
        model,
        "Готов к публикации",
        "datasets/moderation_no_required_output.csv",
    )


def learn_tonality():
    classifier = Classifier(label_map={0: "Негативный", 1: "Позитивный"})

    train_dataset, test_dataset = classifier.load_data("ReviewTonality.csv")

    classifier.train(train_dataset, test_dataset)

    eval_results = classifier.evaluate()

    print("Результаты оценки:", eval_results)
    classifier.save_model("./models/saved_model_tonality")


def check_tonality(data_name="ReviewTonality.csv"):
    model = Classifier(label_map={0: "Негативный", 1: "Позитивный"})
    model.load_model("./models/saved_model_tonality")

    parser = DatasetParser()
    parser.parse_for_model_result(data_name, model, "datasets/tonality_output.csv", column_name='tonality')


def learn_problem():
    df = pd.read_csv("ReviewProblem.csv")
    problem_types = df["label"].unique()
    num_labels = len(problem_types)

    label2id = {label: idx for idx, label in enumerate(problem_types)}
    id2label = {idx: label for label, idx in label2id.items()}

    classifier = Classifier(label_map=id2label)

    train_dataset, test_dataset = classifier.load_data("ReviewProblem.csv")

    classifier.train(train_dataset, test_dataset, num_epochs=10)

    eval_results = classifier.evaluate()

    print("Результаты оценки:", eval_results)
    classifier.save_model("./models/saved_model_problem")

def check_problem(data_name="ReviewProblem.csv"):
    df = pd.read_csv("ReviewProblem.csv")
    problem_types = df["label"].unique()
    num_labels = len(problem_types)

    label2id = {label: idx for idx, label in enumerate(problem_types)}
    id2label = {idx: label for label, idx in label2id.items()}

    model = Classifier(label_map=id2label)
    model.load_model("./models/saved_model_problem")

    parser = DatasetParser()
    parser.parse_for_model_add_column(data_name, model, "datasets/problem_output.csv", column_name='problems')


def all_model_learn():
    print("Learn spam model...")
    learn_spam()
    print("Learn spam model: Done!")
    print("Learn moderation model...")
    learn_moderation()
    print("Learn moderation model: Done!")
    print("Learn tonality model...")
    learn_tonality()
    print("Learn tonality model: Done!")
    print("Learn problem model...")
    learn_problem()
    print("Learn problem model: Done!")


def main_process():
    print("\n\nRemove spam...")
    remove_spam("datasets/review.csv")
    print("Check moderation...")
    check_moderation("datasets/spam_output.csv")
    print("Check tonality...")
    check_tonality("datasets/moderation_no_required_output.csv")
    print("Check problem...")
    check_problem("datasets/tonality_output.csv")
    print("\n\nDone!")


def dataset_review(name):
    print("Starting process!")
    print("Dataset review...")
    parser = DatasetParser()
    parser.dataset_edit(
        name,
        rename={"Unnamed: 1": "text"},
        remove=["Unnamed: 0", "Unnamed: 2"],
        output_file="datasets/review.csv",
    )
    print("Done!")


if __name__ == "__main__":
    directory_path = "./"
    folders = [
        f for f in os.listdir(directory_path) if os.path.isdir(os.path.join("./", f))
    ]

    if "datasets" in folders:
        if not os.path.isfile("./datasets/review.csv"):
            dataset_review("Отзывы (спам, модерация, проблемы, корректные) (1).xlsx")
    else:
        os.mkdir("./datasets")
        dataset_review("Отзывы (спам, модерация, проблемы, корректные) (1).xlsx")
    if "models" in folders:
        if not os.path.isdir("./models/saved_model_spam"):
            print("Learn spam model...")
            learn_spam()
            print("Learn spam model: Done!")
        if not os.path.isdir("./models/saved_model_moderation"):
            print("Learn moderation model...")
            learn_moderation()
            print("Learn moderation model: Done!")
        if not os.path.isdir("./models/saved_model_tonality"):
            print("Learn tonality model...")
            learn_tonality()
            print("Learn tonality model: Done!")
        if not os.path.isdir("./models/saved_model_problem"):
            print("Learn problem model...")
            learn_problem()
            print("Learn problem model: Done!")
    else:
        os.mkdir("./models")
        all_model_learn()

    main_process()
