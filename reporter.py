import numpy as np
import os
import pandas as pd


class Reporter:
    def __init__(self, prefix, indices_list, folds):
        self.prefix = prefix
        self.indices_list = indices_list
        self.folds = folds
        if not os.path.exists("results"):
            os.mkdir("results")
        self.details_columns = [f"Fold-{i}" for i in range(self.folds)]
        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.details = np.zeros((len(self.indices_list), self.folds))
        self.sync_details_file()

    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=["index"], axis=1, inplace=True)
        self.details = df.to_numpy()

    def update_summary(self):
        score_summary = np.zeros((len(self.indices_list), 2))
        for index_index in range(len(self.indices_list)):
            details_row = self.details[index_index]
            details_row = details_row[details_row != 0]
            score_summary[index_index, 0] = np.round(np.mean(details_row),3)
            score_summary[index_index, 1] = np.round(np.std(details_row),3)

        df = pd.DataFrame(data=score_summary, columns=["Mean", "Std"])
        indices_names = ["-".join(indices) for indices in self.indices_list]
        df.insert(0,"index",pd.Series(indices_names))
        df.to_csv(self.summary_file, index=False)

    def set_details(self, index_index, fold_number, acc):
        self.details[index_index,fold_number] = acc

    def get_details(self, index_index, fold_number):
        return self.details[index_index, fold_number]

    def write_details(self):
        details_copy = np.round(self.details, 3)
        df = pd.DataFrame(data=details_copy, columns=self.details_columns)
        indices_names = ["-".join(indices) for indices in self.indices_list]
        df.insert(0,"index",pd.Series(indices_names))
        df.to_csv(self.details_file, index=False)
