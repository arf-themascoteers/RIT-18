from scipy.io import loadmat
import evaluate_rit18
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


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
    cols = [str(i) for i in band_centers]
    cols.append("class")
    rows = []
    total = data.shape[1] * data.shape[2]
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if mask[i,j] != 0:
                row = []
                for k in range(6):
                    row.append(data[k,i,j])
                row.append(labels[i,j])
                rows.append(row)
                now = (i*data.shape[2])+j
                print(f"Row {i} done. {now} done among {total}: {(now/total)*100:.2f}%")
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