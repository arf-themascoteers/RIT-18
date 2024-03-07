import numpy as np
import os
import pandas as pd


class Reporter:
    def __init__(self, prefix, algorithms, folds):
        self.prefix = prefix
        self.algorithms = algorithms
        self.folds = folds
        if not os.path.exists("results"):
            os.mkdir("results")
        self.details_columns = [str(i) for i in range(self.folds)]
        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.details = np.zeros((len(self.algorithms), self.folds))
        self.sync_details_file()


    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=["Algorithm"], axis=1, inplace=True)
        self.details = df.to_numpy()

    def update_summary(self):
        score_summary = np.zeros((self.algorithms, 2))
        for index_algorithm in range(len(self.algorithms)):
            details_row = self.details[index_algorithm]
            score_summary[index_algorithm, 0] = np.round(np.mean(details_row),3)
            score_summary[index_algorithm, 1] = np.round(np.std(details_row),3)

        df = pd.DataFrame(data=score_summary, columns=["Mean", "Std"])
        df.to_csv(self.summary_file, index=False)

    def set_details(self, index_algorithm, fold_number, acc):
        self.details[index_algorithm,fold_number] = acc

    def get_details(self, index_algorithm, fold_number):
        return self.details[index_algorithm, fold_number]

    def write_details(self):
        details_copy = np.round(self.details, 3)
        df = pd.DataFrame(data=details_copy, columns=self.details_columns)
        df.insert(0,"Algorithm",pd.Series(self.algorithms))
        df.to_csv(self.details_file, index=False)
