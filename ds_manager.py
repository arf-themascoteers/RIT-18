import pandas as pd
from sklearn.model_selection import KFold
import torch
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import utils
from pixel_dataset import PixelDataset


class DSManager:
    def __init__(self, folds=10):
        self.folds = folds
        torch.manual_seed(0)
        df = pd.read_csv(utils.get_data_file())
        columns_to_scale = df.columns[:-1]
        scaler = MinMaxScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        self.data = df.sample(frac=1, random_state=1).to_numpy()

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.data)):
            train_data = self.data[train_index]
            train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=1)
            test_data = self.data[test_index]
            train_x = train_data[:, 0:-1]
            train_y = train_data[:, -1]
            test_x = test_data[:, 0:-1]
            test_y = test_data[:, -1]
            validation_x = validation_data[:, 0:-1]
            validation_y = validation_data[:, -1]

            yield PixelDataset(train_x, train_y), \
                PixelDataset(test_x, test_y), \
                PixelDataset(validation_x, validation_y)

    def get_folds(self):
        return self.folds


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dm = DSManager(3,["550","ndvi"])
    for fold_number, (dtrain, dtest, dval) in enumerate(dm.get_k_folds()):
        dataloader = DataLoader(dtrain, batch_size=500, shuffle=False)
        for batch_number, (x, y) in enumerate(dataloader):
            print(x.shape)
            print(y.shape)
            break
        break


    dm = DSManager(3)
    for fold_number, (dtrain, dtest, dval) in enumerate(dm.get_k_folds()):
        dataloader = DataLoader(dtrain, batch_size=500, shuffle=False)
        for batch_number, (x, y) in enumerate(dataloader):
            print(x.shape)
            print(y.shape)
            break
        break