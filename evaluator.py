from networkx import selfloop_edges

from reporter import Reporter
from ds_manager import DSManager
from ann import ANN


class Evaluator:
    def __init__(self, prefix="", folds=10, indices=None):
        self.folds = folds
        self.indices = indices

        if self.indices is None:
            self.indices = []

        self.reporter = Reporter(prefix, self.indices, self.folds)

    def process(self):
        for index_index, index in enumerate(self.indices):
            self.process_index(index_index)

    def process_index(self, index_index):
        index = self.indices[index_index]
        ds = DSManager(self.folds)
        for fold_number, (train_ds, test_ds, validation_ds) in enumerate(ds.get_k_folds()):
            acc = self.reporter.get_details(index_index, fold_number)
            if acc != 0:
                print(f"{fold_number} done already")
                continue
            else:
                print("Start", f"{self.indices[index_index]} - fold {fold_number}")
                acc = self.calculate_score(train_ds, test_ds, validation_ds, index)
                print(f"Accuracy: {acc}")
            self.reporter.set_details(index_index, fold_number, acc)
            self.reporter.write_details()
            self.reporter.update_summary()

    def calculate_score(self, train_ds, test_ds, validation_ds,index):
        ann = ANN(train_ds, test_ds, validation_ds, self.indices[index])
        acc = ann.run()
        return acc

