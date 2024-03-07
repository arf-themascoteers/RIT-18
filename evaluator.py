from reporter import Reporter
from ds_manager import DSManager
from ann_normal_savi import ANNNormalSAVI
from ann_learnable_savi import ANNLearnableSAVI
from ann_learnable_simple_savi import ANNLearnableSimpleSAVI


class Evaluator:
    def __init__(self, prefix="", folds=10, algorithms=None):
        self.folds = folds
        self.algorithms = algorithms

        if self.algorithms is None:
            self.algorithms = ["ann_normal_savi"]

        self.reporter = Reporter(prefix, self.algorithms, self.folds)

    def process(self):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(index_algorithm)

    def process_algorithm(self, index_algorithm):
        algorithm = self.algorithms[index_algorithm]
        ds = DSManager(self.folds)
        for fold_number, (train_ds, test_ds, validation_ds) in enumerate(ds.get_k_folds()):
            acc = self.reporter.get_details(index_algorithm, fold_number)
            if acc != 0:
                print(f"{fold_number} done already")
                continue
            else:
                print("Start", f"{self.algorithms[index_algorithm]} - fold {fold_number}")
                acc = Evaluator.calculate_score(train_ds, test_ds, validation_ds, algorithm)
                print(f"Accuracy: {acc}")
            self.reporter.set_details(index_algorithm, fold_number, acc)
            self.reporter.write_details()
            self.reporter.update_summary()

    @staticmethod
    def calculate_score(train_ds, test_ds, validation_ds,algorithm):
        print(f"Train: {len(train_ds.y)}, Test: {len(test_ds.y)}, Validation: {len(validation_ds.y)}")
        clazz = None
        if algorithm == "ann_normal_savi":
            clazz = ANNNormalSAVI
        elif algorithm == "ann_normal_savi":
            clazz = ANNNormalSAVI
        elif algorithm == "ann_learnable_simple_savi":
            clazz = ANNLearnableSimpleSAVI
        elif algorithm == "ann_learnable_savi":
            clazz = ANNLearnableSAVI

        model_instance = clazz(train_ds, test_ds, validation_ds)
        acc = model_instance.run()
        return acc