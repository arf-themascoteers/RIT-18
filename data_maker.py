from scipy.io import loadmat
import evaluate_rit18
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

grass = [2]
tree = [14]
sand = [15]
water = [16,17]
asphalt = [18]

def syncdf(file, new_rows, cols):
    df = pd.DataFrame(data=new_rows, columns = cols)
    base_df = None
    if not os.path.exists(file):
        base_df = df
    else:
        base_df = pd.read_csv(file)
        base_df = pd.concat([base_df, df], axis=0, ignore_index=True)
    base_df.to_csv(file,index=False)


def make_data(source, task):
    dataset = loadmat(source)
    data_key = f"{task}_data"
    label_key = f"{task}_labels"
    data = dataset[data_key]
    mask = data[-1]
    data = data[:6]
    labels = dataset[label_key]
    file = f"data/{task}.csv"
    band_centers = dataset['band_centers'][0]
    classes = dataset['classes']
    cols = [str(int(i)) for i in band_centers]
    cols.append("class")
    rows = []
    total = int(data.shape[1] * data.shape[2]/400)
    done = 0
    for i in range(0,data.shape[1],20):
        for j in range(0,data.shape[2],20):
            if mask[i,j] != 0:
                row = []
                for k in range(6):
                    row.append(data[k,i,j])
                label = labels[i,j]
                filter_label = -1
                if label in grass:
                    filter_label = 0
                elif label in tree:
                    filter_label = 1
                elif label in sand:
                    filter_label = 2
                elif label in water:
                    filter_label = 3
                elif label in asphalt:
                    filter_label = 4

                if filter_label == -1:
                    continue

                row.append(filter_label)
                rows.append(row)
                done = done + 1
                print(f"Row {i} done. {done} done among {total}: {(done/total)*100:.2f}%")
        if len(rows) > 0:
            syncdf(file, rows, cols)
            rows = []

    # print(band_centers)
    # print(classes)
    # print(len(classes))
    # print(np.unique(mask))
    # print(np.max(data))
    # print(np.unique(labels))


if __name__ == "__main__":
    make_data("rit18_data.mat", "train")