import pandas as pd

df = pd.read_csv("data/train.csv")
df = df.sample(frac=0.0001)
df.to_csv("data/train_min.csv")