from ann_normal_savi import ANNNormalSAVI
from ds_manager import DSManager
import numpy as np

dm = DSManager()
accuracies = []
pcs = []

for fold_number, (train_ds, test_ds, validation_ds) in enumerate(dm.get_k_folds()):
    ann = ANNNormalSAVI(train_ds, test_ds, validation_ds,-0.0215)
    accuracy, pc = ann.run()
    accuracies.append(accuracy)
    pcs.append(accuracy)
    break


accuracies = np.array(accuracies)
pcs = np.abs(np.array(pcs))

print(np.mean(accuracies))
print(np.std(accuracies))
print(accuracies)

print(np.mean(pcs))
print(np.std(pcs))
print(pcs)


