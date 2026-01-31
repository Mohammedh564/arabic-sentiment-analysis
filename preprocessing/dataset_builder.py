import pandas as pd
from arabic_preprocessor import arabicPreprocessor

class datasetBuilder:
    def __init__(self, data_source):
        self.data_source = data_source
        self.data = self.load_data()

    def load_data(self):
        print(f"Loading data from {self.data_source}...")
        return pd.read_csv(self.data_source, sep='\t').sample(frac=1, random_state=42).reset_index(drop=True)

    def preprocess_data(self):
        print("Preprocessing data...")
        preprocessor = arabicPreprocessor()
        self.data['text'] = self.data['text'].apply(preprocessor.preprocess)
        return self.data

    def split_data(self, train_frac=0.8, val_frac=0.1):
        print("Splitting data into train, validation, and test sets...")
        train_end = int(len(self.data) * train_frac)
        val_end = int(len(self.data) * (train_frac + val_frac))

        train_data = self.data[:train_end]
        val_data = self.data[train_end:val_end]
        test_data = self.data[val_end:]

        return train_data, val_data, test_data


# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    DATA_PATH = r"data/raw/ar_reviews_100k.tsv"
    OUTPUT_DIR = r"data/processed/"

    builder = datasetBuilder(DATA_PATH)

    builder.preprocess_data()
    train_df, val_df, test_df = builder.split_data()

    train_df.to_csv(f"{OUTPUT_DIR}train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_DIR}val.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}test.csv", index=False)

    print("Dataset preprocessing and splitting completed successfully.")

