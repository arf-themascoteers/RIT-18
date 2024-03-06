from ann_simple import ANNSimple
from ds_manager import DSManager

dm = DSManager("data/lucas_s2.csv")
r2s = []

for fold_number, (train_ds, test_ds, validation_ds) in enumerate(dm.get_k_folds()):
    ann = ANNSimple(train_ds, test_ds, validation_ds)
    accuracy = ann.run()
    break



