from ann_normal_savi import ANNNormalSAVI
from ds_manager import DSManager

dm = DSManager()
r2s = []

for fold_number, (train_ds, test_ds, validation_ds) in enumerate(dm.get_k_folds()):
    ann = ANNNormalSAVI(train_ds, test_ds, validation_ds)
    accuracy = ann.run()
    break



