from ann_learnable_simple_savi import ANNLearnableSimpleSAVI
from ds_manager import DSManager
import numpy as np

dm = DSManager(folds=3)
accuracies = []
pcs = []

for fold_number, (train_ds, test_ds, validation_ds) in enumerate(dm.get_k_folds()):
    ann = ANNLearnableSimpleSAVI(train_ds, test_ds, validation_ds)
    accuracy = ann.run()
    accuracies.append(accuracy)
    pcs.append(accuracy)


accuracies = np.array(accuracies)

print(np.mean(accuracies))
print(np.std(accuracies))
print(accuracies)


