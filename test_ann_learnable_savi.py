from ann_learnable_savi import ANNLearnableSAVI
from ds_manager import DSManager
import numpy as np

dm = DSManager()
accuracies = []

for fold_number, (train_ds, test_ds, validation_ds) in enumerate(dm.get_k_folds()):
    ann = ANNLearnableSAVI(train_ds, test_ds, validation_ds)
    accuracy = ann.run()
    accuracies.append(accuracy)

accuracies = np.array(accuracies)

print(np.mean(accuracies))
print(np.std(accuracies))
print(accuracies)


