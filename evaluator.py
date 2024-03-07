from reporter import Reporter
from ds_manager import DSManager
from ann import ANN


class Evaluator:
    def __init__(self, prefix="", folds=10, indices=None):
        self.folds = folds
        self.indices_list = indices

        if self.indices_list is None:
            self.indices_list = []

        self.reporter = Reporter(prefix, self.indices_list, self.folds)

    def process(self):
        for indices_index, index in enumerate(self.indices_list):
            self.process_indices(indices_index)

    def process_indices(self, indices_index):
        indices = self.indices_list[indices_index]
        ds = DSManager(self.folds)
        for fold_number, (train_ds, test_ds, validation_ds) in enumerate(ds.get_k_folds()):
            acc = self.reporter.get_details(indices_index, fold_number)
            if acc != 0:
                print(f"{fold_number} done already")
                continue
            else:
                print("Start", f"{self.indices_list[indices_index]} - fold {fold_number}")
                acc = self.calculate_score(train_ds, test_ds, validation_ds, indices)
                print(f"Accuracy: {acc}")
            self.reporter.set_details(indices_index, fold_number, acc)
            self.reporter.write_details()
            self.reporter.update_summary()

    def calculate_score(self, train_ds, test_ds, validation_ds,index):
        ann = ANN(train_ds, test_ds, validation_ds, self.indices_list[index])
        acc = ann.run()
        return acc

