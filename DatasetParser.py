import pandas as pd
import os
from Classifier import Classifier


class DatasetParser:
    def dataset_edit(self, input_file, rename={}, remove=[], output_file="output.csv"):
        filename, file_extension = os.path.splitext(input_file)
        if file_extension == ".csv":
            df = pd.read_csv(input_file)
        elif file_extension == ".xlsx":
            df = pd.read_excel(input_file)
        else:
            return False
        if rename != {}:
            print(rename)
            df = df.rename(columns=rename)
        if remove != []:
            print(remove)
            df = df.drop(columns=remove)
        new_df = pd.DataFrame(df)
        new_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"Новый файл сохранен в: {output_file}")
        return True

    def parse_for_model_slice(self, input_file, model, check, output_file="output.csv"):
        df = pd.read_csv(input_file)
        new_data = []

        for index, row in df.iterrows():
            new_value = row["text"]
            result = model.predict(new_value)
            if result == check:
                new_row = {"text": row["text"]}
                new_data.append(new_row)

        new_df = pd.DataFrame(new_data)
        new_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"Новый файл сохранен в: {output_file}")

    def parse_for_model_result(
        self, input_file, model, output_file="output.csv", column_name="result"
    ):
        df = pd.read_csv(input_file)
        new_data = []

        for index, row in df.iterrows():
            new_value = row["text"]
            result = model.predict(new_value)
            new_row = {"text": row["text"], column_name: result}
            new_data.append(new_row)

        new_df = pd.DataFrame(new_data)
        new_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"Новый файл сохранен в: {output_file}")

    def parse_for_model_add_column(
        self, input_file, model, output_file="output.csv", column_name="result"
    ):
        df = pd.read_csv(input_file)
        new_data = []

        for index, row in df.iterrows():
            text = row["text"]
            result = model.predict(text)
            new_data.append(result)

        df[column_name] = new_data
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"Новый файл сохранен в: {output_file}")


if __name__ == "__main__":
    model = Classifier(label_map={0: "Не спам", 1: "Спам"})
    model.load_model("./saved_model_spam")

    parser = DatasetParser()
    parser.parse_for_model_slice("Review.csv", model, "Не спам")
